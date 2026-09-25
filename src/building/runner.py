"""Running a build's work: the downloads it waits on, and the crops it computes."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import ExitStack, closing
from pathlib import Path

import httpx
from rich.console import Console

from analysis.selector.models.selection import Selection
from building import console as printing
from building import paths, planner
from building.dispatcher import INSTRUMENTS
from building.metadata.index import read_observation_metadata, write_index
from building.metadata.observation import (
    ObservationMetadata,
    observation_metadata,
)
from building.models.job import Job, Outcome, Plan
from building.models.progress import Progress
from building.models.settings import Settings
from building.preprocessing.common import store
from building.scheduler import Scheduler
from common.console import print_failure
from common.fetch.http import TLS_CONTEXT

CHECKPOINT_BYTES = 100 * 1024**3


def build_dataset(
    settings: Settings,
    picked: Sequence[Selection],
    force: bool = False,
    checkpoint: Callable[[], None] | None = None,
) -> int:
    """Build one dataset over the tiles it is handed.

    Args:
        settings: The settled choices for the build, naming the dataset.
        picked: The tiles to build, each with the observations its window keeps.
        force: Whether to build every crop again, rather than only the missing ones.
        checkpoint: What publishes the dataset so far, or None for a local run.

    Returns:
        code: A process exit code, non zero when any product failed to build.
    """
    console = Console()
    started_at = time.monotonic()
    outcomes = run_build(
        settings,
        picked,
        console,
        paths.dataset_root(settings.name),
        force=force,
        checkpoint=checkpoint,
    )
    printing.print_summary(outcomes, time.monotonic() - started_at, console)
    return 1 if any(one.error for one in outcomes) else 0


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
        checkpoint: What publishes the dataset as it stands, or None.

    Returns:
        collected: Every finished outcome, in completion order.
    """
    # Reused, so a run pays for a connection once a host rather than once a file
    connections = sum(settings.downloads.values()) * 2
    limits = httpx.Limits(
        max_connections=connections, max_keepalive_connections=connections
    )
    with httpx.Client(limits=limits, verify=TLS_CONTEXT) as ode:
        try:
            indexed = read_observation_metadata(root)
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
        printing.describe(plan, settings, console)
        progress = Progress(len(plan.jobs))
        # A download waits on the network and a build on the cores, so the pools differ.
        with (
            ProcessPoolExecutor(max_workers=settings.workers) as building,
            ExitStack() as pools,
            printing.watch(progress),
        ):
            fetching = {
                archive: pools.enter_context(ThreadPoolExecutor(max_workers=downloads))
                for archive, downloads in settings.downloads.items()
            }
            held = Scheduler(
                ode, fetching, building, build_product, root, settings, progress
            ).outcomes(plan.jobs)
            with closing(held) as outcomes:
                if checkpoint is not None:
                    outcomes = _checkpointed(outcomes, plan, root, checkpoint)
                collected = printing.render(
                    outcomes, len(plan.jobs), "building", console
                )
    write_index(plan, collected, root, on_disk=checkpoint is None)
    return collected


def _checkpointed(
    outcomes: Iterator[Outcome],
    plan: Plan,
    root: Path,
    checkpoint: Callable[[], None],
) -> Iterator[Outcome]:
    """Hand on every outcome, publishing what is built once its crops fill the disk.

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
    held = 0
    for outcome in outcomes:
        collected.append(outcome)
        yield outcome
        # A crop an earlier checkpoint already sent is no longer on disk to weigh.
        held += sum(
            (root / one.path).stat().st_size
            for one in outcome.records
            if (root / one.path).exists()
        )
        # The last crops are published by the run itself, so they wait here.
        if held < CHECKPOINT_BYTES:
            continue
        held = 0
        try:
            # An index is written first, so what is published is readable on its own.
            write_index(plan, collected, root, on_disk=False)
            checkpoint()
        except Exception as error:  # noqa: BLE001
            # A checkpoint is insurance: a build outlives one it could not write.
            failed += 1
            print_failure("the checkpoint", error, failed)


def build_product(job: Job, root: Path) -> Outcome:
    """Cut one downloaded product to every tile that kept it, and write each.

    Args:
        job: The product to build, and the tiles to cut it to.
        root: The dataset's own root directory.

    Returns:
        outcome: The outcome, its written samples and the first error a cut raised.

    Raises:
        Exception: Whatever reading the product raised, collected as a failure.
    """
    steps = INSTRUMENTS[job.instrument]
    written: list[ObservationMetadata] = []
    missed = 0
    failed: Exception | None = None
    try:
        # Read once however many tiles want it, which is why the product is the unit.
        observation = steps.read_observation(job.identifier)
        for frame in job.frames:
            try:
                held = steps.crop(observation, frame)
            except Exception as error:  # noqa: BLE001
                # A tile failing to cut is kept as the error, and the rest still cut.
                failed = failed or error
                continue
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
    return Outcome(job, records=tuple(written), missed=missed, error=failed)
