"""Everything the pipeline prints: plans, live progress, and totals."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress

from analysis.models.job import Plan
from analysis.models.progress import CoverageSummary, DownloadSummary, ProgressEvent
from analysis.models.settings import Settings

# How many items are named before the rest are counted
LISTED = 5

# Set by a platform run, whose log takes plain flushed lines rather than a bar.
PLAIN_LOG_ENV = "PIPELINE_PLAIN_LOG"

# How many progress lines a stage prints where no cursor can be moved
LOGGED_LINES = 50


def describe(
    download: Plan, coverage: Plan, settings: Settings, console: Console
) -> None:
    """Print what each half of the run has to do before it starts.

    Args:
        download: The plan produced by the download planner.
        coverage: The plan produced by the coverage planner.
        settings: The settled choices for the run, which size both halves.
        console: The console to print on.
    """
    console.print(
        f"download: {download.feature_count} features x {download.set_count} sets, "
        f"{len(download.jobs)} to run, {download.skipped_existing} already "
        f"downloaded, {settings.workers} workers"
    )
    console.print(
        f"coverage: {coverage.feature_count} features, "
        f"{coverage.set_count} instrument sets, {len(coverage.jobs)} to compute, "
        f"{coverage.skipped_existing} already done, "
        f"{settings.workers} workers x {settings.union_threads} threads"
    )


def render(
    events: Iterable[ProgressEvent], total: int, description: str, console: Console
) -> None:
    """Draw a live progress bar while consuming runner events.

    Args:
        events: The progress events produced by a runner.
        total: The number of units in the run.
        description: The label for the progress task.
        console: The console to render on.
    """
    # A platform log takes plain flushed lines, since no cursor can be moved there
    if os.environ.get(PLAIN_LOG_ENV):
        step = max(1, total // LOGGED_LINES)
        for event in events:
            outcome = event.outcome
            if outcome.failed:
                print(f"error {outcome.label}: {outcome.error}", flush=True)
            if event.completed % step == 0 or event.completed == total:
                _reached(description, event.completed, total, outcome.label)
        return
    with Progress(
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(description, total=total)
        for event in events:
            if event.outcome.failed:
                console.print(
                    f"[red]error[/red] {event.outcome.label}: {event.outcome.error}"
                )
            progress.update(task, completed=event.completed)


def logged(description: str) -> Callable[[int, int], None]:
    """Return a progress callback printing how far a stage has got.

    Args:
        description: The label for the stage, carried on every line printed.

    Returns:
        progress: A callback taking how many units are done and how many there are.
    """

    def moved(done: int, total: int) -> None:
        """Print where the stage has reached, on the units it reports on."""
        if done % max(1, total // LOGGED_LINES) == 0 or done == total:
            _reached(description, done, total)

    return moved


def print_interrupted() -> None:
    """Print the notice shown when a run is stopped with Ctrl-C."""
    Console().print(
        "[yellow]interrupted: pending jobs cancelled, finished files kept. "
        "Re-run to resume.[/yellow]"
    )


def print_summary(
    download: DownloadSummary,
    coverage: CoverageSummary,
    indexed: int,
    missing: Sequence[Path],
    console: Console,
) -> None:
    """Print the totals for a finished run.

    Args:
        download: The download half's totals.
        coverage: The coverage half's totals.
        indexed: Summary rows gathered into the catalogue index.
        missing: The instrument sets that still have no artifact on disk.
        console: The console to print on.
    """
    console.print(
        f"downloaded {download.ran} sets, {download.failed} failed, "
        f"in {download.elapsed:.1f}s"
    )
    console.print(
        f"computed {coverage.computed} sets, {coverage.events:,} observation rows, "
        f"{coverage.failed} failed, {indexed:,} rows indexed, "
        f"in {coverage.elapsed:.1f}s"
    )
    if coverage.empty or coverage.discarded:
        console.print(
            f"[yellow]{coverage.empty} sets measured nothing, "
            f"{coverage.discarded:,} records discarded for no footprint, "
            f"no start time, or no overlap[/yellow]"
        )
    if not missing:
        return
    console.print(f"[yellow]{len(missing)} sets still have no artifact:[/yellow]")
    for source in missing[:LISTED]:
        console.print(f"[yellow]  {source}[/yellow]")
    if len(missing) > LISTED:
        console.print(f"[yellow]  and {len(missing) - LISTED} more[/yellow]")


def _reached(description: str, completed: int, total: int, label: str = "") -> None:
    """Print how far a stage has got, in the plain form a platform log takes.

    Args:
        description: The label for the stage.
        completed: How many units are finished.
        total: How many there are.
        label: What just finished, where the stage names its units.
    """
    share = completed / total
    print(
        f"{description} {completed}/{total} ({share:.0%}) {label}".rstrip(), flush=True
    )
