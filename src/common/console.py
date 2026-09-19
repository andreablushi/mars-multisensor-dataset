"""What both halves print the same way, on a terminal or into a platform's log."""

from __future__ import annotations

import os
from collections.abc import Sequence

from rich.console import Console

# Set by a platform run, whose log takes plain flushed lines rather than a bar.
PLAIN_LOG_ENV = "PIPELINE_PLAIN_LOG"

# How many items are named before the rest are counted
LISTED = 5

# How many failures a run names as it hits them, the summary counting them all
LOGGED_ERRORS = 50


def plain_log() -> bool:
    """Say whether a run prints flushed lines rather than drawing a bar.

    Returns:
        plain: True on a platform, whose log has no cursor to move.
    """
    return bool(os.environ.get(PLAIN_LOG_ENV))


def reached(description: str, completed: int, total: int, label: str = "") -> None:
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


def named_failure(label: str, error: BaseException, counted: int) -> None:
    """Name one failure as a run hits it, until too many have been named.

    Args:
        label: What failed.
        error: What it raised.
        counted: How many have failed so far, this one counted.
    """
    if counted <= LOGGED_ERRORS:
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


def print_interrupted(kept: str) -> None:
    """Print the notice shown when a run is stopped with Ctrl-C.

    Args:
        kept: What a stopped run leaves behind, which each half names its own.
    """
    Console().print(
        f"[yellow]interrupted: pending jobs cancelled, {kept} kept. "
        "Re-run to resume.[/yellow]"
    )
