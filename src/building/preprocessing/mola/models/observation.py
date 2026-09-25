"""One MOLA grid as it comes off disk, its sheets found but none of their bins read."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class MolaObservation:
    """One grid, before any of its bins are read.

    Attributes:
        identifier: The grid as `configs.GRIDS` names it, which crops are stored under.
        resolution: How many bins of the grid one degree holds.
        files: The image and label of every product that landed, by sheet.
        polar: Whether it is projected onto a pole rather than split into sheets.
    """

    identifier: str
    resolution: int
    files: dict[str, Path]
    polar: bool
