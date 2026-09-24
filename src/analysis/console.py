"""Everything the pipeline prints: plans, live progress, and totals."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress

from analysis.models.job import Outcome, Plan
from common import console as printing

# How many progress lines a stage prints where no cursor can be moved
LOGGED_LINES = 50


def describe(download: Plan, coverage: Plan, console: Console) -> None:
    """Print what each half of the run has to do before it starts.

    Args:
        download: The download jobs still to run.
        coverage: The coverage jobs still to run.
        console: The console to print on.
    """
    for stage, plan in (("download", download), ("coverage", coverage)):
        console.print(f"{stage}: {len(plan.jobs)} to run, {plan.skipped} already done")


def report(stage: str, done: int, total: int, label: str = "") -> None:
    """Print how far a stage has got, once every fiftieth of it.

    Args:
        stage: The stage, carried on every line printed.
        done: How many units are finished.
        total: How many there are.
        label: What just finished, where the stage names its units.
    """
    if done % max(1, total // LOGGED_LINES) == 0 or done == total:
        printing.reached(stage, done, total, label)


class Tracker:
    """One stage's progress, drawn as a bar or logged where no cursor can move."""

    def __init__(self, stage: str, total: int, console: Console) -> None:
        """Set up the stage's bar, left undrawn on a plain log.

        Args:
            stage: The stage, labelling the bar or every line.
            total: How many jobs it runs.
            console: The console to draw on.
        """
        self.stage, self.total, self.console = stage, total, console
        self.done = self.failed = 0
        self.bar = Progress(
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            console=console,
            # A platform log takes plain flushed lines, since no cursor can move there
            disable=printing.plain_log(),
        )
        self.task = self.bar.add_task(stage, total=total)

    def __enter__(self) -> Tracker:
        """Start drawing the bar."""
        self.bar.start()
        return self

    def __exit__(self, *raised: object) -> None:
        """Stop drawing the bar."""
        self.bar.stop()

    def advance(self, outcome: Outcome) -> None:
        """Count one finished job, naming it when it failed.

        Args:
            outcome: The job that just finished.
        """
        self.done += 1
        if outcome.failed:
            self.failed += 1
            printing.named_failure(
                outcome.label, outcome.error, self.failed, self.console
            )
        self.bar.update(self.task, completed=self.done)
        if self.bar.disable:
            report(self.stage, self.done, self.total, outcome.label)


def print_summary(
    downloads: Sequence[Outcome],
    measured: Sequence[Outcome],
    elapsed: float,
    missing: Sequence[Path],
    console: Console,
) -> None:
    """Print the totals for a finished run.

    Args:
        downloads: Every finished download.
        measured: Every finished coverage job.
        elapsed: How long the two halves took, in seconds.
        missing: The instrument sets that still have no artifact on disk.
        console: The console to print on.
    """
    succeeded = [outcome for outcome in measured if not outcome.failed]
    empty = sum(1 for outcome in succeeded if not outcome.events)
    discarded = sum(outcome.discarded for outcome in measured)
    download_failed = sum(outcome.failed for outcome in downloads)
    console.print(
        f"download: {len(downloads) - download_failed} done, {download_failed} failed"
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
