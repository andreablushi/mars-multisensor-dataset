"""One MOLA grid as it comes off disk, its sheets found but none of their bins read."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from building.configs.mola import Grid


@dataclass(frozen=True, slots=True)
class MolaObservation:
    """One grid, before any of its bins are read.

    Attributes:
        identifier: The grid as `configs.GRIDS` names it.
        grid: How fine it is, and whether it is projected onto a pole.
        files: The image and label of every product that landed, by sheet.
    """

    identifier: str
    grid: Grid
    files: dict[str, Path]
