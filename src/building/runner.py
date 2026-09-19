"""Running a build's work: the downloads it waits on, and the crops it computes."""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import closing
from functools import partial
from pathlib import Path

import httpx
from rich.console import Console

from analysis.selector.models.selection import Selection
from building import console as printing
from building import paths, planner
from building.dispatcher import INSTRUMENTS
from building.metadata import read as metadata_read
from building.metadata import write as metadata
from building.metadata.observation import (
    ObservationMetadata,
    observation_metadata,
)
from building.models.budget import Budget, memory_bytes
from building.models.job import Job, Outcome, Plan
from building.models.progress import (
    BUILDING,
    FETCHING,
    HOLDING,
    QUEUED,
    Progress,
)
from building.models.settings import Settings
from building.preprocessing.common import store
from common.console import named_failure

# How much of the box's memory a build may hold. The rest is not spare: the box is
# charged for the downloads in flight and for the cache of every file written and
# read back, none of which this budget meters.
MEMORY_SHARE = 0.45

CHECKPOINT_PRODUCTS = 500


def run_build(
    settings: Settings,
    picked: Sequence[Selection],
    console: Console,
    root: Path,
    *,
    force: bool = False,
    checkpoint: Callable[[], None] | None = None,
) -> list[Outcome]:
    """Fetch every product a build needs and cut each to the tiles that kept it.

    Args:
        settings: The settled choices for the build.
        picked: The tiles to build, each with the observations its window keeps.
        console: The console to render on.
        root: The directory this build of the dataset is written in.
        force: Whether to rebuild crops that are already written.
        checkpoint: What publishes the dataset as it stands and may take its
            crops off disk, so a run that dies is resumed from what it left and
            every crop its index names counts as built, or None to publish only
            at the end.

    Returns:
        collected: Every finished outcome, in completion order.
    """
    # Every core builds, and what each build holds is measured as it lands.
    budget = Budget(int(memory_bytes() * MEMORY_SHARE))
    # Reused, so a run pays for a connection once a host rather than once a file
    limits = httpx.Limits(
        max_connections=settings.downloads * 2,
        max_keepalive_connections=settings.downloads * 2,
    )
    with httpx.Client(limits=limits) as ode:
        try:
            indexed = metadata_read.read_observation_metadata(root)
            named = frozenset(one.path for one in indexed)
        except FileNotFoundError:
            named = frozenset()
        # A crop the index cannot name is unreadable, so it is built again.
        if not force:
            dropped = 0
            for path in paths.crop_paths(root):
                if str(path.relative_to(root)) not in named:
                    path.unlink()
                    dropped += 1
            if dropped:
                console.print(f"dropping {dropped:,} crops the index does not name")
        published = named if checkpoint else frozenset()
        plan = planner.build_plan(picked, root, ode, force=force, published=published)
        printing.describe(plan, settings, budget, console)
        progress = Progress(len(plan.jobs))
        # A download waits on the network and a build on the cores, so the pools differ.
        with (
            ProcessPoolExecutor(max_workers=settings.workers) as building,
            # A thread waiting on memory holds no download back, so this is wider
            ThreadPoolExecutor(max_workers=settings.in_flight) as fetching,
            printing.watch(progress),
        ):
            held = _outcomes(
                plan.jobs, ode, fetching, building, root, settings, budget, progress
            )
            with closing(held) as outcomes:
                if checkpoint is not None:
                    outcomes = _checkpointed(outcomes, plan, root, checkpoint)
                collected = printing.render(
                    outcomes, len(plan.jobs), "building", console
                )
    _indexed(plan, collected, root, on_disk=checkpoint is None)
    return collected


def _checkpointed(
    outcomes: Iterator[Outcome],
    plan: Plan,
    root: Path,
    checkpoint: Callable[[], None],
) -> Iterator[Outcome]:
    """Hand on every outcome, publishing what is built every so many products.

    Args:
        outcomes: The outcomes as the runner finishes them.
        plan: What the build set out to do, whose tiles the index covers.
        root: The dataset's own root directory.
        checkpoint: What publishes the dataset as it stands.

    Yields:
        outcome: Each outcome as it came in, unchanged.
    """
    collected: list[Outcome] = []
    failed = 0
    for outcome in outcomes:
        collected.append(outcome)
        yield outcome
        # The last products are published by the run itself, so they wait here.
        if len(collected) % CHECKPOINT_PRODUCTS:
            continue
        try:
            # An index is written first, so what is published is readable on its own.
            _indexed(plan, collected, root, on_disk=False)
            checkpoint()
        except Exception as error:  # noqa: BLE001
            # A checkpoint is insurance: a build outlives one it could not write.
            failed += 1
            named_failure("the checkpoint", error, failed)


def _outcomes(
    jobs: tuple[Job, ...],
    ode: httpx.Client,
    fetching: ThreadPoolExecutor,
    building: ProcessPoolExecutor,
    root: Path,
    settings: Settings,
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
        settings: The settled choices for the build, which bound how many
            products run at once and how many of them are downloading.
        budget: The memory the builds running at once share between them.
        progress: What every product still in the build is doing.

    Yields:
        outcome: One outcome per job, in the order they finish.
    """
    finished: queue.Queue[Outcome] = queue.Queue()
    # A place to land in, and a turn on the network, since neither bounds the other.
    waiting = threading.Semaphore(settings.in_flight)
    downloading = threading.Semaphore(settings.downloads)

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
    """Cut one downloaded product to every tile that kept it, and write each.

    Args:
        job: The product to build, and the tiles to cut it to.
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
        # Read once however many tiles want it, which is why the product is the unit.
        observation = steps.read_observation(job.identifier)
        for frame in job.frames:
            try:
                held = steps.crop(observation, frame)
            except Exception as error:  # noqa: BLE001
                # What is on disk is handed back, so no written sample misses the index.
                return Outcome(job, records=tuple(written), error=error)
            # Reaching none of a tile is no failure, coverage being a box overlap.
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
                )
            )
    finally:
        # A product goes once every tile that wanted it is cut; it is a cache.
        if steps.discard:
            steps.discard(job.identifier)
    return Outcome(job, records=tuple(written), missed=missed)


def _indexed(
    plan: Plan,
    collected: Sequence[Outcome],
    root: Path,
    *,
    on_disk: bool,
) -> None:
    """Write the index over every crop of the tiles covered, not this run's alone.

    Args:
        plan: What the build set out to do, whose tiles alone the index names, so a
            tile the build no longer covers leaves it.
        collected: What every job of this run left.
        root: The dataset's own root directory.
        on_disk: Whether an earlier run's record is kept only while its crop is
            on disk, rather than for as long as the index names it.
    """
    written = [held for one in collected for held in one.records]
    rewritten = {one.identity for one in written}
    tiles = {one.identity: one for one in plan.tiles}
    try:
        standing = metadata_read.read_observation_metadata(root)
    except FileNotFoundError:
        standing = []
    # What an earlier run left of the tiles covered, less what this run rewrote and
    # what has been deleted.
    records = [
        one
        for one in standing
        if one.tile in tiles
        and one.identity not in rewritten
        and (not on_disk or (root / one.path).exists())
    ] + written
    # What the dataset holds, which is every instrument in it and not a wish.
    held = tuple(sorted({one.instrument for one in records}))
    grids = {
        name: INSTRUMENTS[name].layout.band_centres_nm
        for name in held
        if name in INSTRUMENTS and INSTRUMENTS[name].layout.band_centres_nm
    }
    metadata.write_metadata(list(tiles.values()), records, held, grids, root)
