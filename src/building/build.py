"""Building one dataset here, from the tiles it is handed."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

from rich.console import Console

from analysis.selector.models.selection import Selection
from building import console, paths, runner
from building.models.settings import Settings


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
    printing = Console()
    started_at = time.monotonic()
    outcomes = runner.run_build(
        settings,
        picked,
        printing,
        paths.dataset_root(settings.name),
        force=force,
        checkpoint=checkpoint,
    )
    console.print_summary(outcomes, time.monotonic() - started_at, printing)
    return 1 if any(one.error for one in outcomes) else 0
