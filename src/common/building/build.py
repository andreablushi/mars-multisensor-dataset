"""Building one dataset here, from the tiles it is handed."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

from rich.console import Console

from common.analysis.selector.models.selection import Selection
from common.building import console, paths, runner
from common.building.configs import overall


def build_dataset(
    name: str,
    picked: Sequence[Selection],
    force: bool = False,
    workers: int | None = None,
    checkpoint: Callable[[], None] | None = None,
) -> int:
    """Build one dataset over the tiles it is handed.

    Args:
        name: What the dataset is called, the directory it is written in.
        picked: The tiles to build, each with the observations its window keeps.
        force: Whether to build every crop again, rather than only the missing ones.
        workers: How many products to build at once, or None for the config.
        checkpoint: What publishes the dataset as it stands, for a platform run
            that is resumed from what it left, and None for a run here.

    Returns:
        code: A process exit code, non zero when any product failed to build.
    """
    choices = overall.load(name, workers=workers)
    printing = Console()
    started_at = time.monotonic()
    outcomes = runner.run_build(
        choices,
        picked,
        printing,
        paths.dataset_root(name),
        force=force,
        checkpoint=checkpoint,
    )
    console.print_summary(outcomes, time.monotonic() - started_at, printing)
    return 1 if any(one.error for one in outcomes) else 0
