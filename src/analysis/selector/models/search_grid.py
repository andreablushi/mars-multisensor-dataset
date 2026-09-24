"""The grid a tile is searched over, and which of its cells it really covers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SearchGrid:
    """One tile's grid, and what a window over it is measured against.

    Attributes:
        cells: How many cells the grid holds.
        area_km2: How much ground the tile really has inside it.
        cell_km2: How much ground one cell of the grid covers.
        inside: Which cells of the grid the tile really covers.
    """

    cells: int
    area_km2: float
    cell_km2: float
    inside: frozenset[int]
