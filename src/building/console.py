"""Everything a build prints: its plan, live progress, and totals."""

from __future__ import annotations

import threading
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn
from rich.progress import Progress as Bar

from building.models import budget as memory
from building.models.budget import Budget
from building.models.job import Outcome, Plan
from building.models.progress import Progress
from building.models.settings import Settings
from shared import console as printing

# Progress lines where no cursor moves; the platform keeps a run's first 100 kB
LOGGED_LINES = 100

# How often a run says what it is doing, so a stalled build does not look slow
WATCHED_SECONDS = 300.0

# What one gibibyte is, which the memory a run holds is said in
GIB = 1024**3


def describe(plan: Plan, settings: Settings, budget: Budget, console: Console) -> None:
    """Print what a build has to do before it starts.

    Args:
        plan: What the planner worked out.
        settings: The settled choices for the build, which size it.
        budget: The memory those builds share, which settles how many of the
            heaviest products run at once.
        console: The console to print on.
    """
    crops = sum(len(job.frames) for job in plan.jobs)
    console.print(
        f"building {len(plan.features)} features from {len(plan.jobs)} products, "
        f"{crops} crops to write, {plan.skipped_existing} already written, "
        f"{plan.unread} kept observations no instrument here reads, "
        f"{plan.crowded} features left out over {settings.max_observations} "
        f"observations"
    )
    console.print(
        f"instruments: {', '.join(sorted({job.instrument for job in plan.jobs}))}; "
        f"share {settings.share:.0%}, seed {settings.seed}; "
        f"build pool {settings.workers}, download pool {settings.downloads}, "
        f"{settings.in_flight} products may wait, "
        f"{budget.total / GIB:.0f} GiB between them"
    )


@contextmanager
def watch(progress: Progress) -> Iterator[None]:
    """Say what the build is doing every so often while it runs.

    Args:
        progress: What every product still in the build is doing.
    """
    # A moving bar already says a run is alive; only a flat log needs telling.
    if not printing.plain_log():
        yield
        return
    done = threading.Event()

    def said() -> None:
        """Print what the build is doing until it is over."""
        while not done.wait(WATCHED_SECONDS):
            print(progress.standing, flush=True)

    watcher = threading.Thread(target=said, daemon=True)
    watcher.start()
    try:
        yield
    finally:
        done.set()
        watcher.join()


def _high_water() -> str:
    """Return the most memory the box has held, to read against what it was given.

    Returns:
        held: The high water mark to print, and an empty string where nothing
            counts one, so a run outside a container says nothing of it.
    """
    peak = memory.peak_bytes()
    return f", peak {peak / 1024**3:.1f} GiB" if peak else ""


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
        collected: Every outcome collected, in completion order.
    """
    collected: list[Outcome] = []
    # A platform log takes plain flushed lines, since no cursor can be moved there
    if printing.plain_log():
        step = max(1, total // LOGGED_LINES)
        failed = 0
        for outcome in outcomes:
            collected.append(outcome)
            if outcome.error:
                failed += 1
                printing.named_failure(outcome.job.label, outcome.error, failed)
            if len(collected) % step == 0 or len(collected) == total:
                # The one named is the one just finished, never the one under way
                printing.reached(
                    description,
                    len(collected),
                    total,
                    f"{outcome.job.label} done{_high_water()}",
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
    printing.print_listed([f"{one.job.label}: {one.error}" for one in failed], console)
