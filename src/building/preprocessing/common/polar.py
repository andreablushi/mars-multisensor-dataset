"""The projection a tile at a pole is read on: stereographic metres from its centre."""

from __future__ import annotations

import numpy as np

from building.preprocessing.common.crop import marked, sample_sizes, sampled
from building.preprocessing.common.models.relative_position import (
    PolarGrid,
    RelativePosition,
)
from shared.maths import geodesy
from shared.maths.geodesy import TURN
from shared.models.tile import Tile

# The longest segment the box is walked in, a chord leaving its arc by under a pixel.
STEP = 0.1

# How many samples of a cut are crossed at once, since a grid can be huge.
BLOCK = 4_000_000


def cut(
    down: np.ndarray, across: np.ndarray, grid: PolarGrid, frame: Tile, span: float
) -> tuple[tuple[np.ndarray, ...], np.ndarray | None] | None:
    """Return which bins of one grid projected onto a pole the box keeps.

    Args:
        down: The northing of every line, in the grid's own metres.
        across: The easting of every sample, in the same metres.
        grid: The grid the two are measured on.
        frame: The tile's local frame, carrying the box the catalogue gives it.
        span: How many degrees of longitude that box covers.

    Returns:
        bounds: The lines and the samples to keep.
        inside: Which of them truly falls in the box, and None where every one
            of them does.
    """
    ring = geodesy.stereographic_forward(
        *geodesy.bbox_ring(
            frame.min_lat, frame.max_lat, frame.west_lon, frame.east_lon, STEP
        ),
        *grid,
    )
    # The box projects to a sector, and the ring its edge traces bounds it.
    lines = np.flatnonzero((down >= ring[1].min()) & (down <= ring[1].max()))
    samples = np.flatnonzero((across >= ring[0].min()) & (across <= ring[0].max()))
    if not lines.size or not samples.size:
        return None
    # Only the sector's rectangle is crossed back, a block of its lines at a time.
    inside = np.empty((lines.size, samples.size), dtype=bool)
    sizes = (lines.size, samples.size)
    for block, lon, lat in sampled(
        down[lines], across[samples], True, grid, sizes, BLOCK
    ):
        inside[block] = (
            (lat >= frame.min_lat)
            & (lat <= frame.max_lat)
            & ((lon - frame.west_lon) % TURN <= span)
        )
    if not inside.any():
        return None
    return (lines, samples), marked(inside)


def placed(
    down: np.ndarray,
    across: np.ndarray,
    separable: bool,
    frame: Tile,
    grid: PolarGrid | None,
    place: PolarGrid,
) -> RelativePosition:
    """Return where the samples one cut keeps sit, in the metres of the tile's pole.

    Args:
        down: The latitude of every line it keeps in degrees, or its northing in
            the metres of `grid`, one per sample where the two are not separable.
        across: The longitude of every sample it keeps, or its easting, holding
            the same.
        separable: Whether those two hold one axis each rather than a value for
            every sample.
        frame: The tile's local frame, whose centre the offsets stand from.
        grid: The grid the samples sit on, and None where they are degrees.
        place: The grid the tile is read on, which every instrument shares.

    Returns:
        position: The metres north and east of that centre, separable only where
            the samples sit on a grid of the same pole already.
    """
    centre_x, centre_y = geodesy.stereographic_forward(
        frame.centre_lon, frame.centre_lat, *place
    )
    if grid == place:
        return RelativePosition(
            down - float(centre_y), across - float(centre_x), separable, place
        )
    # Two grids of one pole differ by their sphere alone, which is a scale.
    if grid is not None and grid[:2] == place[:2]:
        scale = place[2] / grid[2]
        return RelativePosition(
            down * scale - float(centre_y),
            across * scale - float(centre_x),
            separable,
            place,
        )
    sizes = sample_sizes(down, across, separable)
    north = np.empty(sizes, dtype=float)
    east = np.empty(sizes, dtype=float)
    for block, lon, lat in sampled(down, across, separable, grid, sizes, BLOCK):
        x, y = geodesy.stereographic_forward(lon, lat, *place)
        north[block] = y - float(centre_y)
        east[block] = x - float(centre_x)
    return RelativePosition(north, east, False, place)
