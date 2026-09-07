"""Everything a build prints: its plan, live progress, and totals."""

from __future__ import annotations

import os
import threading
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn
from rich.progress import Progress as Bar

from building.models.budget import Budget
from building.models.job import Outcome, Plan
from building.models.progress import Progress
from building.models.settings import Settings

# How many items are named before the rest are counted
LISTED = 5

# Set by a platform run, whose log takes plain flushed lines rather than a bar.
PLAIN_LOG_ENV = "PIPELINE_PLAIN_LOG"

# How many progress lines a stage prints where no cursor can be moved
LOGGED_LINES = 2000

# How often a run says what it is doing, so a build that has stopped moving
# says so rather than looking the same as one that is merely slow
WATCHED_SECONDS = 300.0

# What one gibibyte is, which the memory a run holds is said in
GIB = 1024**3


def describe(
    plan: Plan,
    settings: Settings,
    pools: tuple[int, int, int],
    budget: Budget,
    console: Console,
) -> None:
    """Print what a build has to do before it starts.

    Args:
        plan: What the planner worked out.
        settings: The settled choices for the build, which size it.
        pools: The builds, the downloads and the products that may wait, as the
            runner worked them out from the machine.
        budget: The memory those builds share, which settles how many of the
            heaviest products run at once.
        console: The console to print on.

    Returns:
        None.
    """
    crops = sum(len(job.frames) for job in plan.jobs)
    building, fetching, ready = pools
    console.print(
        f"building {len(plan.features)} features from {len(plan.jobs)} products, "
        f"{crops} crops to write, {plan.skipped_existing} already written, "
        f"{plan.unread} kept observations no instrument here reads"
    )
    console.print(
        f"instruments: {', '.join(sorted({job.instrument for job in plan.jobs}))}; "
        f"share {settings.share:.0%}, seed {settings.seed}; "
        f"build pool {building}, download pool {fetching}, "
        f"{ready} products may wait, {budget.total / GIB:.0f} GiB between them"
    )


@contextmanager
def watch(progress: Progress) -> Iterator[None]:
    """Say what the build is doing every so often while it runs.

    Args:
        progress: What every product still in the build is doing.

    Yields:
        None, for as long as the build runs.
    """
    # A moving bar already says a run is alive; only a flat log needs telling.
    if not os.environ.get(PLAIN_LOG_ENV):
        yield
        return
    done = threading.Event()

    def said() -> None:
        """Print what the build is doing until it is over.

        Returns:
            None.
        """
        while not done.wait(WATCHED_SECONDS):
            print(progress.standing, flush=True)

    watcher = threading.Thread(target=said, daemon=True)
    watcher.start()
    try:
        yield
    finally:
        done.set()
        watcher.join()


def render(
    outcomes: Iterable[Outcome], total: int, description: str, console: Console
) -> list[Outcome]:
    """Draw a live progress bar while collecting what each job left.

    Args:
        outcomes: The outcomes as the runner finishes them.
        total: How many jobs there are.
        description: The label for the progress task.
        console: The console to render on.

    Returns:
        Every outcome collected, in completion order.
    """
    collected: list[Outcome] = []
    # A platform log takes plain flushed lines, since no cursor can be moved there
    if os.environ.get(PLAIN_LOG_ENV):
        step = max(1, total // LOGGED_LINES)
        for outcome in outcomes:
            collected.append(outcome)
            if outcome.error:
                print(f"error {outcome.job.label}: {outcome.error}", flush=True)
            if len(collected) % step == 0 or len(collected) == total:
                share = len(collected) / total
                print(
                    f"{description} {len(collected)}/{total} ({share:.0%}) "
                    f"{outcome.job.label}",
                    flush=True,
                )
        return collected
    with Bar(
        BarColumn(bar_width=None), MofNCompleteColumn(), console=console
    ) as progress:
        task = progress.add_task(description, total=total)
        for outcome in outcomes:
            collected.append(outcome)
            if outcome.error:
                console.print(f"[red]error[/red] {outcome.job.label}: {outcome.error}")
            progress.update(task, completed=len(collected))
    return collected


def print_summary(
    outcomes: Sequence[Outcome], elapsed: float, console: Console
) -> None:
    """Print the totals for a finished build.

    Args:
        outcomes: What every job left.
        elapsed: How long the build took, in seconds.
        console: The console to print on.

    Returns:
        None.
    """
    written = sum(len(one.records) for one in outcomes)
    missed = sum(one.missed for one in outcomes)
    failed = [one for one in outcomes if one.error]
    console.print(
        f"built {len(outcomes) - len(failed)} products into {written:,} crops, "
        f"{len(failed)} failed, in {elapsed:.1f}s"
    )
    if missed:
        console.print(
            f"[yellow]{missed:,} crops came out empty, the product reaching "
            f"none of the feature it was kept for[/yellow]"
        )
    if not failed:
        return
    console.print(f"[yellow]{len(failed)} products failed:[/yellow]")
    for one in failed[:LISTED]:
        console.print(f"[yellow]  {one.job.label}: {one.error}[/yellow]")
    if len(failed) > LISTED:
        console.print(f"[yellow]  and {len(failed) - LISTED} more[/yellow]")


def print_interrupted() -> None:
    """Print the notice shown when a build is stopped with Ctrl-C.

    Returns:
        None.
    """
    Console().print(
        "[yellow]interrupted: pending jobs cancelled, written crops kept. "
        "Re-run to resume.[/yellow]"
    )
