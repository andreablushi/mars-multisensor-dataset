"""Running a build's work: the downloads it waits on, and the crops it computes."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

import httpx
from rich.console import Console

from analysis.selector.models.selection import Selection
from building import console as printing
from building import paths, planner
from building.metadata.index import read_observation_metadata, write_index
from building.models.job import Outcome, Plan
from building.models.progress import Progress
from building.models.settings import Settings
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
    root = paths.dataset_root(settings.name)
    named = indexed_crops(root, console, force=force)
    published = named if checkpoint else frozenset()
    # Reused, so a run pays for a connection once a host rather than once a file
    connections = sum(settings.downloads.values()) * 2
    limits = httpx.Limits(
        max_connections=connections, max_keepalive_connections=connections
    )
    with httpx.Client(limits=limits, verify=TLS_CONTEXT) as ode:
        plan = planner.build_plan(picked, root, force=force, published=published)
        printing.print_plan(plan, settings, console)
        outcomes = build_outcomes(plan, settings, root, ode, console, checkpoint)
    write_index(plan, outcomes, root, on_disk=checkpoint is None)
    printing.print_summary(outcomes, time.monotonic() - started_at, console)
    return 1 if any(one.error for one in outcomes) else 0


def indexed_crops(root: Path, console: Console, *, force: bool) -> frozenset[str]:
    """Return the crops the index names, deleting every other one unless forced.

    Args:
        root: The directory this build of the dataset is written in.
        console: The console to report the deleted crops on.
        force: Whether every crop is built again, so none is deleted.

    Returns:
        named: The relative path of every crop the index names.
    """
    named = frozenset(one.path for one in read_observation_metadata(root))
    if force:
        return named
    # A crop the index cannot name is unreadable, so it is built again.
    dropped = 0
    for path in paths.crop_paths(root):
        if str(path.relative_to(root)) not in named:
            path.unlink()
            dropped += 1
    if dropped:
        console.print(f"dropping {dropped:,} crops the index does not name")
    return named


def build_outcomes(
    plan: Plan,
    settings: Settings,
    root: Path,
    ode: httpx.Client,
    console: Console,
    checkpoint: Callable[[], None] | None,
) -> list[Outcome]:
    """Fetch every product a plan needs and cut each to the tiles that kept it.

    Args:
        plan: What the build has to do.
        settings: The settled choices for the build, which size its pools.
        root: The directory this build of the dataset is written in.
        ode: The client every download is asked through.
        console: The console to render on.
        checkpoint: What publishes the dataset as it stands, or None.

    Returns:
        collected: Every finished outcome, in completion order.
    """
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
        scheduler = Scheduler(ode, fetching, building, root, settings, progress)
        outcomes = scheduler.outcomes(plan.jobs)
        if checkpoint is not None:
            outcomes = _checkpointed(outcomes, plan, root, checkpoint)
        return printing.collect_outcomes(outcomes, len(plan.jobs), console)


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
