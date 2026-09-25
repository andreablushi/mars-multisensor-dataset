"""What one grid of the gridded record landed as, which a tile's box is read from."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class MolaObservation:
    """What one grid holds, before any of its bins are read.

    Attributes:
        name: The grid as `configs.GRIDS` names it, which its crops are stored under.
        resolution: How many bins of the grid one degree holds.
        files: The image and label of every product that landed, by sheet.
        polar: Whether it is projected onto a pole rather than split into sheets.
    """

    name: str
    resolution: int
    files: dict[str, Path]
    polar: bool = False
