"""What both halves print the same way, on a terminal or into a platform's log."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

# Set by a platform run, whose log takes plain flushed lines rather than a bar.
PLAIN_LOG_ENV = "PIPELINE_PLAIN_LOG"

# How many items are named before the rest are counted
LISTED = 5

# How many failures a run names as it hits them, the summary counting them all
LOGGED_ERRORS = 50


def plain_log() -> bool:
    """Say whether a run prints flushed lines, as on a platform, rather than a bar."""
    return bool(os.environ.get(PLAIN_LOG_ENV))


def print_progress_line(
    description: str, completed: int, total: int, label: str, lines: int
) -> None:
    """Print how far a stage has got, in the plain form a platform log takes.

    Args:
        description: The label for the stage.
        completed: How many units are finished.
        total: How many there are.
        label: What just finished, where the stage names its units.
        lines: About how many lines the whole stage prints.
    """
    if completed % max(1, total // lines) and completed != total:
        return
    share = completed / total
    print(
        f"{description} {completed}/{total} ({share:.0%}) {label}".rstrip(), flush=True
    )


def print_failure(
    label: str, error: BaseException, counted: int, console: Console | None = None
) -> None:
    """Name one failure as a run hits it, a plain log naming only the first few.

    Args:
        label: What failed.
        error: What it raised.
        counted: How many have failed so far, this one counted.
        console: The console a bar is drawn on, printed on in red off a plain log.
    """
    if console is not None and not plain_log():
        console.print(f"[red]error[/red] {label}: {error}")
    elif counted <= LOGGED_ERRORS:
        print(f"error {label}: {error}", flush=True)
    elif counted == LOGGED_ERRORS + 1:
        print("the summary counts the failures from here", flush=True)


def print_listed(lines: Sequence[str], console: Console) -> None:
    """Print the first few lines a summary has to name, and count the rest.

    Args:
        lines: What there is to name, in the order to name it.
        console: The console to print on.
    """
    for line in lines[:LISTED]:
        console.print(f"[yellow]  {line}[/yellow]")
    if len(lines) > LISTED:
        console.print(f"[yellow]  and {len(lines) - LISTED} more[/yellow]")


class Tracker:
    """One stage's progress, drawn as a bar or logged where no cursor can move."""

    def __init__(
        self,
        stage: str,
        total: int,
        console: Console,
        lines: int,
        logged: Callable[[str], str] = str,
    ) -> None:
        """Set up the stage's bar, left undrawn on a plain log.

        Args:
            stage: The stage, labelling the bar or every line.
            total: How many jobs it runs.
            console: The console to draw on.
            lines: About how many lines the stage logs where no cursor can move.
            logged: What a logged line names a finished job by, given its label.
        """
        self.stage, self.total, self.console = stage, total, console
        self.lines, self.logged = lines, logged
        self.done = self.failed = 0
        self.bar = Progress(
            TextColumn("{task.description}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            console=console,
            # A platform log takes plain flushed lines, since no cursor can move there
            disable=plain_log(),
        )
        self.task = self.bar.add_task(stage, total=total)

    def __enter__(self) -> Tracker:
        """Start drawing the bar."""
        self.bar.start()
        return self

    def __exit__(self, *raised: object) -> None:
        """Stop drawing the bar."""
        self.bar.stop()

    def advance(self, label: str, error: BaseException | None) -> None:
        """Count one finished job, naming it when it failed.

        Args:
            label: What the job was.
            error: What it raised, or None where it succeeded.
        """
        self.done += 1
        if error is not None:
            self.failed += 1
            print_failure(label, error, self.failed, self.console)
        self.bar.update(self.task, completed=self.done)
        if self.bar.disable:
            print_progress_line(
                self.stage, self.done, self.total, self.logged(label), self.lines
            )
