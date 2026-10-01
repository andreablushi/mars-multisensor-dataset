"""The group of tiles ODE is asked about at once, and the box that bounds them."""

from __future__ import annotations

from dataclasses import dataclass

from common.maths.box import Boxed
from common.models.tile import Tile


@dataclass(frozen=True, slots=True)
class TileGroup(Boxed):
    """A run of neighbouring tiles, queried and measured together.

    Attributes:
        name: What tells this group from every other, such as "r05_s12".
        tiles: The tiles it holds, band by band and west to east.
        min_lat: The southernmost latitude any of them reaches, in degrees.
        max_lat: The northernmost latitude any of them reaches, in degrees.
        west_lon: The westernmost longitude any of them reaches, 0 to 360.
        east_lon: The easternmost longitude, the westernmost where it circles a pole.
    """

    name: str
    tiles: tuple[Tile, ...]
    min_lat: float
    max_lat: float
    west_lon: float
    east_lon: float
