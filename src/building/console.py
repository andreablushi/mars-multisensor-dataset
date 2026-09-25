"""Everything a build prints: its plan, live progress, and totals."""

from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager, suppress
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn
from rich.progress import Progress as Bar

from building.models.job import Outcome, Plan
from building.models.progress import Progress
from building.models.settings import Settings
from common import console as printing

# Progress lines where no cursor moves; the platform keeps a run's first 100 kB
LOGGED_LINES = 100

# How often a run says what it is doing, so a stalled build does not look slow
WATCHED_SECONDS = 300.0

DESCRIPTION = "building"

# Where a container writes the most memory it has held, by cgroup version.
CGROUP_PEAKS = (
    Path("/sys/fs/cgroup/memory.peak"),
    Path("/sys/fs/cgroup/memory/memory.max_usage_in_bytes"),
)


def print_plan(plan: Plan, settings: Settings, console: Console) -> None:
    """Print what a build has to do before it starts.

    Args:
        plan: What the planner worked out.
        settings: The settled choices for the build, which size it.
        console: The console to print on.
    """
    crops = sum(len(job.frames) for job in plan.jobs)
    console.print(
        f"building {len(plan.tiles)} tiles from {len(plan.jobs)} products, "
        f"{crops} crops to write, {plan.skipped_existing} already written, "
        f"{plan.unread} kept observations no instrument here reads"
    )
    console.print(
        f"instruments: {', '.join(sorted({job.instrument for job in plan.jobs}))}; "
        f"built as {settings.name}; "
        f"build pool {settings.workers}, download pools "
        f"{', '.join(f'{name} {n}' for name, n in settings.downloads.items())}, "
        f"each archive holding {settings.workers} more that wait on a core"
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
    watcher = threading.Thread(
        target=_print_standing, args=(progress, done), daemon=True
    )
    watcher.start()
    try:
        yield
    finally:
        done.set()
        watcher.join()


def _print_standing(progress: Progress, done: threading.Event) -> None:
    """Print what the build is doing every so often until it is over.

    Args:
        progress: What every product still in the build is doing.
        done: What is set once the build is over.
    """
    while not done.wait(WATCHED_SECONDS):
        print(progress.standing, flush=True)


def _memory_peak() -> str:
    """Return the most memory the box has held, to read against what it was given.

    Returns:
        peak: The high water mark to print, or empty where nothing counts one.
    """
    counted = list(CGROUP_PEAKS)
    with suppress(OSError):
        # A container reads its own cgroup as the root, and a host process does not
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            if line.startswith("0::"):
                own = line.removeprefix("0::").strip().lstrip("/")
                counted.append(Path("/sys/fs/cgroup") / own / "memory.peak")
    for path in counted:
        try:
            return f", peak {int(path.read_text().split()[0]) / 1024**3:.1f} GiB"
        except (OSError, ValueError):
            continue
    return ""


def collect_outcomes(
    outcomes: Iterable[Outcome], total: int, console: Console
) -> list[Outcome]:
    """Collect what each job left, drawing how far the build has got.

    Args:
        outcomes: The outcomes as the runner finishes them.
        total: How many jobs there are.
        console: The console to render on.

    Returns:
        collected: Every outcome collected, in completion order.
    """
    collected: list[Outcome] = []
    failed = 0
    # A platform log takes plain flushed lines, since no cursor can be moved there
    plain = printing.plain_log()
    step = max(1, total // LOGGED_LINES)
    columns = (BarColumn(bar_width=None), MofNCompleteColumn())
    with Bar(*columns, console=console, disable=plain) as progress:
        task = progress.add_task(DESCRIPTION, total=total)
        for outcome in outcomes:
            collected.append(outcome)
            if outcome.error:
                failed += 1
                printing.print_failure(
                    outcome.job.label, outcome.error, failed, console
                )
            progress.update(task, completed=len(collected))
            if plain and (len(collected) % step == 0 or len(collected) == total):
                # The one named is the one just finished, never the one under way
                printing.print_progress_line(
                    DESCRIPTION,
                    len(collected),
                    total,
                    f"{outcome.job.label} done{_memory_peak()}",
                )
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
            f"none of the tile it was kept for[/yellow]"
        )
    if not failed:
        return
    console.print(f"[yellow]{len(failed)} products failed:[/yellow]")
    printing.print_listed([f"{one.job.label}: {one.error}" for one in failed], console)
    lacking: dict[str, set[str]] = defaultdict(set)
    for one in failed:
        written = {record.tile for record in one.records}
        lacking[one.job.instrument].update(
            frame.name for frame in one.job.frames if frame.name not in written
        )
    incomplete = set().union(*lacking.values())
    console.print(
        f"[yellow]{len(incomplete):,} tiles lack a product, "
        + ", ".join(
            f"{name} on {len(held):,}" for name, held in sorted(lacking.items())
        )
        + "; a rerun without --force fills them in[/yellow]"
    )
