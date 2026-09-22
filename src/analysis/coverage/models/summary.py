"""The one row describing what an instrument set covered of a tile."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class Summary:
    """One row of the per-instrument-set coverage summary.

    Attributes:
        tile: The tile's name, such as "b123_c0456".
        set_key: The instrument set the records were asked for by.
        ihid: The instrument host identifier.
        iid: The instrument identifier.
        pt: The product type.
        tile_area_km2: The area of the tile's bounding box.
        covered_km2: How much of it the set reached.
        covered_frac: The same as a share of the tile.
        n_obs: How many observations the row covers.
        t_first: When the earliest of them started.
        t_last: When the latest of them started.
        span_days: How long the row's observations span.
        mask_cells: How many of the tile's grid cells fall inside it.
        pixels: How many pixels the set landed inside the tile, revisits counted.
        grid_side: How many cells the tile's grid holds along each axis.
        cell_km2: How much ground one cell of that grid covers.
        grid_mask: Which cells of that grid fall inside the tile, packed as a mask.
    """

    tile: str
    set_key: str
    ihid: str
    iid: str
    pt: str
    tile_area_km2: float
    covered_km2: float
    covered_frac: float
    n_obs: int
    t_first: datetime
    t_last: datetime
    span_days: float
    mask_cells: int
    pixels: float
    grid_side: int
    cell_km2: float
    grid_mask: bytes
