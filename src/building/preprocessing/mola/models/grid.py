"""The tiles of one grid that landed, which a feature's box is merged from."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class MolaGrid:
    """What one grid of the gridded record holds, before any of it is read.

    Attributes:
        name: The grid, as `configs.GRIDS` names it, which is also what every
            crop merged from it is stored under.
        resolution: How many bins of the grid one degree holds.
        files: The image of every tile of it that landed, keyed by tile, each
            with its own label beside it.
    """

    name: str
    resolution: int
    files: dict[str, Path]
