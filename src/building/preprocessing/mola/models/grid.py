"""What one grid of the record landed as, which a feature's box is read from."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class MolaGrid:
    """What one grid holds, before any of its bins are read.

    Attributes:
        name: The grid, as `configs.GRIDS` names it, which is also what every
            crop read from it is stored under.
        resolution: How many bins of the grid one degree holds.
        files: The image of every product of it that landed, keyed by the tile
            or the product it is, each with its own label beside it.
        polar: Whether it is projected onto a pole rather than tiled in
            longitude and latitude.
    """

    name: str
    resolution: int
    files: dict[str, Path]
    polar: bool = False
