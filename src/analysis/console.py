"""Everything the pipeline prints: plans, live progress, and totals."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from rich.console import Console

from analysis.models.job import Outcome, Plan
from common import console as printing

# How many progress lines a stage prints where no cursor can be moved
LOGGED_LINES = 50


def print_plans(download: Plan, coverage: Plan, console: Console) -> None:
    """Print what each half of the run has to do before it starts.

    Args:
        download: The download jobs still to run.
        coverage: The coverage jobs still to run.
        console: The console to print on.
    """
    for stage, plan in (("download", download), ("coverage", coverage)):
        console.print(f"{stage}: {len(plan.jobs)} to run, {plan.skipped} already done")


def print_progress(stage: str, done: int, total: int, label: str = "") -> None:
    """Print how far a stage has got, once every fiftieth of it."""
    printing.print_progress_line(stage, done, total, label, LOGGED_LINES)


def print_summary(
    downloaded: Sequence[Outcome],
    measured: Sequence[Outcome],
    elapsed: float,
    missing: Sequence[Path],
    console: Console,
) -> None:
    """Print the totals for a finished run.

    Args:
        downloaded: Every finished download.
        measured: Every finished coverage job.
        elapsed: How long the two halves took, in seconds.
        missing: The metadata files that still have no coverage summary on disk.
        console: The console to print on.
    """
    succeeded = [outcome for outcome in measured if not outcome.failed]
    empty = sum(1 for outcome in succeeded if not outcome.events)
    discarded = sum(outcome.discarded for outcome in measured)
    download_failed = sum(outcome.failed for outcome in downloaded)
    console.print(
        f"download: {len(downloaded) - download_failed} done, {download_failed} failed"
    )
    console.print(
        f"coverage: {len(succeeded) - empty} done, "
        f"{len(measured) - len(succeeded)} failed, "
        f"{sum(outcome.events for outcome in succeeded):,} observation rows"
    )
    console.print(f"download and coverage took {elapsed:.1f}s")
    if empty or discarded:
        console.print(
            f"[yellow]{empty} sets measured nothing, "
            f"{discarded:,} records discarded for no footprint, "
            f"no start time, or no overlap[/yellow]"
        )
    if missing:
        console.print(f"[yellow]{len(missing)} sets still have no artifact:[/yellow]")
        printing.print_listed([str(source) for source in missing], console)
