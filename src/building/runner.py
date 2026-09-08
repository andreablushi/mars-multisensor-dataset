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
from building.models.budget import Budget, memory_bytes
from building.models.job import Job, Outcome, Plan
from building.models.progress import BUILDING, FETCHING, HOLDING, QUEUED, Progress
from building.models.settings import Settings
from building.preprocessing.common import store

# How much of the box's memory a build may hold, the rest left to everything else.
MEMORY_SHARE = 0.7

# How many downloads run at once; they wait on an archive and not on the cores
DOWNLOADS = 48


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
        collected: Every finished outcome, in completion order.

    Raises:
        FileNotFoundError: When no selection has been written to build from.
    """
    # Every core builds, and what each build holds is measured as it lands.
    building_count = max(1, settings.cores or os.cpu_count() or 1)
    fetching_count = DOWNLOADS
    # Enough waiting to feed every builder while every download is still in flight.
    ready = building_count + fetching_count
    budget = Budget(int(memory_bytes() * MEMORY_SHARE))
    # Reused, so a run pays for a connection once a host rather than once a file
    held = httpx.Limits(
        max_connections=fetching_count * 2,
        max_keepalive_connections=fetching_count * 2,
    )
    with httpx.Client(limits=held) as ode:
        plan = planner.build_plan(settings, root, ode, force=force)
        printing.describe(
            plan, settings, (building_count, fetching_count, ready), budget, console
        )
        progress = Progress(len(plan.jobs))
        # A download waits on the network and a build on the cores, so the pools differ.
        with (
            ProcessPoolExecutor(max_workers=building_count) as building,
            # A thread waiting on memory holds no download back, so this is wider
            ThreadPoolExecutor(max_workers=ready) as fetching,
            printing.watch(progress),
        ):
            held = _outcomes(
                plan.jobs,
                ode,
                fetching,
                building,
                root,
                (fetching_count, ready),
                budget,
                progress,
            )
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
    counts: tuple[int, int],
    budget: Budget,
    progress: Progress,
) -> Iterator[Outcome]:
    """Fetch every product and build it the moment there is room, in whatever order.

    Args:
        jobs: The products to fetch and build, heaviest first.
        ode: The client every download is asked through.
        fetching: The threads the downloads run on.
        building: The processes the builds run on.
        root: The dataset's own root directory.
        counts: How many downloads may run at once, and how many downloaded
            products may wait at once to be built.
        budget: The memory the builds running at once share between them.
        progress: What every product still in the build is doing.

    Yields:
        outcome: One outcome per job, in the order they finish.
    """
    fetching_count, ready = counts
    finished: queue.Queue[Outcome] = queue.Queue()
    # A place to land in, and a turn on the network, since neither bounds the other.
    waiting = threading.Semaphore(ready)
    downloading = threading.Semaphore(fetching_count)

    def finish(outcome: Outcome, ticket: object, held: int) -> None:
        """Record what one job left and give back everything it took.

        Args:
            outcome: What the job left, whether it was built or failed.
            ticket: What the job was tracked by while it was still in the build.
            held: How much memory it was holding, and zero where it held none.
        """
        if held:
            budget.release(held)
        progress.left(ticket, finished=True)
        finished.put(outcome)
        waiting.release()

    def fetched(job: Job) -> None:
        """Bring one product down, take the memory it needs, and hand it on.

        Args:
            job: The product to fetch.
        """
        # The place is taken before the download, so the room is never given elsewhere.
        waiting.acquire()
        steps = INSTRUMENTS[job.instrument]
        ticket, held = progress.entered(job.label, QUEUED), 0
        try:
            with downloading:
                progress.moved(ticket, FETCHING)
                steps.fetch(job.identifier, ode)
            # Only now is there a product to measure, and so a share to ask for.
            progress.moved(ticket, HOLDING)
            held = budget.acquire(
                steps.held_bytes(job.identifier)
                if steps.held_bytes
                else steps.worker_bytes
            )
            progress.moved(ticket, BUILDING)
            building.submit(build_product, job, root).add_done_callback(
                partial(built, job, ticket, held)
            )
        except Exception as error:  # noqa: BLE001
            # The download failed or the pool is closing, so this builds nowhere.
            finish(Outcome(job, error=error), ticket, held)

    def built(job: Job, ticket: object, held: int, done: Future[Outcome]) -> None:
        """Record what one job's build left, a worker the pool lost included.

        Args:
            job: The job that was built.
            ticket: What the job was tracked by while it was still in the build.
            held: How much memory it was holding while it built.
            done: What the build pool left.
        """
        try:
            outcome = done.result()
        except Exception as error:  # noqa: BLE001
            outcome = Outcome(job, error=error)
        finish(outcome, ticket, held)

    # Every path leaves one outcome and gives its place back, or it waits for ever.
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
        outcome: The outcome, holding the record of every sample written, and the error
            that stopped it where one did after some were already on disk.

    Raises:
        Exception: Whatever reading the product off disk raised, which the pool
            hands back for the runner to collect as this job's own failure.
    """
    steps = INSTRUMENTS[job.instrument]
    written: list[ObservationMetadata] = []
    missed = 0
    try:
        # Read once however many features want it, which is why the product is the unit.
        observation = steps.read_observation(job.identifier)
        for frame in job.frames:
            try:
                held = steps.crop(observation, frame)
            except Exception as error:  # noqa: BLE001
                # What is on disk is handed back, so no written sample misses the index.
                return Outcome(job, records=tuple(written), error=error)
            # Reaching none of a feature is no failure, coverage being a box overlap.
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
        # A product goes once every feature that wanted it is cut; it is a cache.
        if steps.discard:
            steps.discard(job.identifier)
    return Outcome(job, records=tuple(written), missed=missed)


def _indexed(
    plan: Plan, collected: Sequence[Outcome], settings: Settings, root: Path
) -> None:
    """Write the index over every crop the dataset holds, not this run's alone.

    Args:
        plan: What the build set out to do, whose features this run covers.
        collected: What every job of this run left.
        settings: The settled choices for the build.
        root: The dataset's own root directory.
    """
    written = [held for one in collected for held in one.records]
    rewritten = {one.identity for one in written}
    try:
        standing = metadata_read.read_observation_metadata(root)
        earlier = metadata_read.read_feature_metadata(root)
    except FileNotFoundError:
        standing, earlier = [], {}
    # What an earlier run left, less what this run rewrote and what has been deleted.
    records = [
        one
        for one in standing
        if one.identity not in rewritten and (root / one.path).exists()
    ] + written
    features = {one.identity: one for one in plan.features}
    # A feature this run missed is carried forward, so no record names an unknown one.
    for one in records:
        if one.feature not in features and one.feature in earlier:
            features[one.feature] = earlier[one.feature]
    # What the dataset holds, which is every instrument in it and not a wish.
    held = tuple(sorted({one.instrument for one in records}))
    metadata.write_metadata(list(features.values()), records, held, root)
