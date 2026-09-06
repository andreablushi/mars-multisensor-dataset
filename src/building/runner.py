"""Running a build's work: the downloads it waits on, and the crops it computes."""

from __future__ import annotations

import os
import queue
import threading
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import closing
from functools import partial
from pathlib import Path

import httpx
from rich.console import Console

from building import console as printing
from building import planner
from building.dispatcher import INSTRUMENTS
from building.metadata import read as metadata_read
from building.metadata import write as metadata
from building.metadata.observation import ObservationMetadata, observation_metadata
from building.models.job import Job, Outcome, Plan
from building.models.settings import Settings
from building.preprocessing.common import store

# How much of what the machine has free a build may hold at once, the rest left
# to the downloads, to the page cache the reads go through, and to everything
# else running beside it.
MEMORY_SHARE = 0.7

# Where a container writes the memory it is held to, which is what a job on a
# platform was given rather than what the machine it landed on has.
CGROUP_LIMITS = (
    Path("/sys/fs/cgroup/memory.max"),
    Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
)

# How many downloads run per build, and the most that run at all. A download
# waits on an archive rather than on the processor, so more of them run than
# there are builds, up to what one archive is worth asking at once.
FETCHING_PER_BUILD = 4
MOST_FETCHING = 16

# How many downloaded products may wait per build, which has to be well above
# one or a build waits on the network it was meant to be running ahead of.
READY_PER_BUILD = 4


def _room() -> int:
    """Return how much memory this run may hold, in bytes.

    Returns:
        The share of what is free that a build is allowed, measured against the
        limit a container holds it to where there is one.
    """
    free = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    for path in CGROUP_LIMITS:
        try:
            free = min(free, int(path.read_text().split()[0]))
        except (OSError, ValueError):
            continue
    return int(free * MEMORY_SHARE)


def pools(instruments: tuple[str, ...], cores: int | None) -> tuple[int, int, int]:
    """Return how many builds run, how many downloads, and how many may wait.

    A build holds a whole product in memory, and the largest of them is a CTX
    scan many times the size of anything else, so what the machine has room for
    and not what it has cores for is what sizes the pool.

    Args:
        instruments: The instruments the build covers, which say how much one
            build of it holds.
        cores: How many cores the run was given, or None for the machine's.

    Returns:
        The builds to run at once, the downloads to run at once, and how many
        downloaded products may wait for a build to reach them.
    """
    held = max(
        INSTRUMENTS[name].worker_bytes for name in instruments if name in INSTRUMENTS
    )
    building = max(1, min(cores or os.cpu_count() or 1, _room() // held))
    return (
        building,
        min(MOST_FETCHING, building * FETCHING_PER_BUILD),
        (building * READY_PER_BUILD),
    )


def run_build(
    settings: Settings,
    console: Console,
    root: Path,
    *,
    force: bool = False,
) -> list[Outcome]:
    """Fetch every product a build needs and cut each to the features that kept it.

    Args:
        settings: The settled choices for the build.
        console: The console to render on.
        root: The directory this build of the dataset is written in.
        force: Whether to rebuild crops that are already written.

    Returns:
        Every finished outcome, in completion order.

    Raises:
        FileNotFoundError: When no selection has been written to build from.
    """
    building_count, fetching_count, ready = pools(settings.instruments, settings.cores)
    with httpx.Client() as ode:
        plan = planner.build_plan(settings, root, ode, force=force)
        printing.describe(
            plan, settings, (building_count, fetching_count, ready), console
        )
        # A download waits on the network and a build waits on the processor, so
        # the two run on pools of their own and neither waits for the other.
        with (
            ProcessPoolExecutor(max_workers=building_count) as building,
            ThreadPoolExecutor(max_workers=fetching_count) as fetching,
        ):
            held = _outcomes(plan.jobs, ode, fetching, building, root, ready)
            with closing(held) as outcomes:
                collected = printing.render(
                    outcomes, len(plan.jobs), "building", console
                )
    _indexed(plan, collected, settings, root)
    return collected


def _outcomes(
    jobs: tuple[Job, ...],
    ode: httpx.Client,
    fetching: ThreadPoolExecutor,
    building: ProcessPoolExecutor,
    root: Path,
    ready: int,
) -> Iterator[Outcome]:
    """Fetch every product and build it the moment it lands, in whatever order.

    Args:
        jobs: The products to fetch and build, heaviest first.
        ode: The client every download is asked through.
        fetching: The threads the downloads run on.
        building: The processes the builds run on.
        root: The dataset's own root directory.
        ready: How many downloaded products may wait at once to be built.

    Yields:
        One outcome per job, in the order they finish.
    """
    finished: queue.Queue[Outcome] = queue.Queue()
    waiting = threading.Semaphore(ready)

    def finish(outcome: Outcome) -> None:
        """Record what one job left and give back the place it took.

        Args:
            outcome: What the job left, whether it was built or failed.

        Returns:
            None.
        """
        finished.put(outcome)
        waiting.release()

    def fetched(job: Job) -> None:
        """Bring one product down and hand it to the pool that builds it.

        Args:
            job: The product to fetch.

        Returns:
            None.
        """
        # The place is taken before the download rather than after, so the room
        # a product is about to need is never given away to another one.
        waiting.acquire()
        try:
            INSTRUMENTS[job.instrument].fetch(job.identifier, ode)
            building.submit(build_product, job, root).add_done_callback(
                partial(built, job)
            )
        except Exception as error:  # noqa: BLE001
            # The download failed, or the build pool is shutting down, and
            # either way this job is built nowhere.
            finish(Outcome(job, error=error))

    def built(job: Job, done: Future[Outcome]) -> None:
        """Record what one job's build left, a worker the pool lost included.

        Args:
            job: The job that was built.
            done: What the build pool left.

        Returns:
            None.
        """
        try:
            outcome = done.result()
        except Exception as error:  # noqa: BLE001
            outcome = Outcome(job, error=error)
        finish(outcome)

    # Every path leaves exactly one outcome on the queue and gives back the one
    # place it took, since a job leaving neither would be waited on for ever.
    for job in jobs:
        fetching.submit(fetched, job)
    for _ in jobs:
        yield finished.get()


def build_product(job: Job, root: Path) -> Outcome:
    """Cut one downloaded product to every feature that kept it, and write each.

    Args:
        job: The product to build, and the features to cut it to.
        root: The dataset's own root directory.

    Returns:
        The outcome, holding the record of every sample written, and the error
        that stopped it where one did after some were already on disk.

    Raises:
        Exception: Whatever reading the product off disk raised, which the pool
            hands back for the runner to collect as this job's own failure.
    """
    steps = INSTRUMENTS[job.instrument]
    written: list[ObservationMetadata] = []
    missed = 0
    try:
        # Read and cleaned once however many features want it, which is what
        # makes the product rather than the feature the unit of work.
        observation = steps.read_observation(job.identifier)
        for frame in job.frames:
            try:
                held = steps.crop(observation, frame)
            except Exception as error:  # noqa: BLE001
                # What is already on disk is handed back, so a later failure
                # never leaves a written sample out of the index.
                return Outcome(job, records=tuple(written), error=error)
            # A product reaching none of a feature is no failure: the coverage
            # it was kept for is a box overlap, and a crop can come out empty.
            if held is None:
                missed += 1
                continue
            path = store.write_sample(held, steps.layout, frame, root)
            written.append(
                observation_metadata(
                    held,
                    frame,
                    steps.layout,
                    str(path.relative_to(root)),
                    t_start=job.t_start,
                    altitude=steps.altitude(held) if steps.altitude else None,
                )
            )
    finally:
        # The tree is a cache, so a product is gone the moment every feature
        # that wanted it has been cut, which is what holds a build to the room
        # its downloads were given rather than to everything it ever fetched.
        steps.discard(job.identifier)
    return Outcome(job, records=tuple(written), missed=missed)


def _indexed(
    plan: Plan, collected: Sequence[Outcome], settings: Settings, root: Path
) -> None:
    """Write the index over every crop the dataset holds, not this run's alone.

    Args:
        plan: What the build set out to do, whose features this run covers.
        collected: What every job of this run left.
        settings: The settled choices for the build, which name its instruments.
        root: The dataset's own root directory.

    Returns:
        None.
    """
    written = [held for one in collected for held in one.records]
    rewritten = {one.identity for one in written}
    try:
        standing = metadata_read.read_observation_metadata(root)
        earlier = metadata_read.read_feature_metadata(root)
    except FileNotFoundError:
        standing, earlier = [], {}
    # What an earlier run left, less what this run rewrote and what has since
    # been deleted from the tree.
    records = [
        one
        for one in standing
        if one.identity not in rewritten and (root / one.path).exists()
    ] + written
    features = {one.identity: one for one in plan.features}
    # A feature this run did not cover is carried forward with the records an
    # earlier run left of it, so no record names a feature nothing describes.
    for one in records:
        if one.feature not in features and one.feature in earlier:
            features[one.feature] = earlier[one.feature]
    metadata.write_metadata(
        list(features.values()), records, settings.instruments, settings.version, root
    )
