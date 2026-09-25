"""The projection a tile at a pole is read on: stereographic metres from its centre."""

from __future__ import annotations

from functools import partial

import numpy as np

from building.preprocessing.common import geometry
from building.preprocessing.common.models.position import Position
from common.maths import geodesy
from common.maths.geodesy import TURN, PolarGrid
from common.models.tile import Tile

# The longest segment the box is walked in, a chord leaving its arc by under a pixel.
STEP = 0.1


def cut(
    position: Position, frame: Tile, span: float
) -> tuple[tuple[np.ndarray, ...], np.ndarray | None] | None:
    """Return which bins of one grid projected onto a pole the box keeps.

    Args:
        position: The samples, placed in the metres of the grid they sit on.
        frame: The tile's local frame, carrying the box the catalogue gives it.
        span: How many degrees of longitude that box covers.

    Returns:
        bounds: The bins to keep of each ground axis, outermost first.
        inside: Which kept bins truly fall in the box, or None for all.
        None: Where the grid reaches none of the box.
    """
    grid = position.grid
    ring_x, ring_y = geodesy.stereographic_forward(
        *geodesy.bbox_ring(
            frame.min_lat, frame.max_lat, frame.west_lon, frame.east_lon, STEP
        ),
        *grid,
    )
    # The box projects to a sector, and the ring its edge traces bounds it.
    lines = np.flatnonzero(
        (position.north >= ring_y.min()) & (position.north <= ring_y.max())
    )
    across = np.flatnonzero(
        (position.east >= ring_x.min()) & (position.east <= ring_x.max())
    )
    if not lines.size or not across.size:
        return None
    # A latitude is a radius here, so the box keeps one band of the sector alone.
    band = sorted(
        abs(float(geodesy.stereographic_forward(grid[0], lat, *grid)[1]))
        for lat in (frame.min_lat, frame.max_lat)
    )
    # A south grid runs its eastings the other way round, so its turn does too.
    sign = -1.0 if grid[1] else 1.0
    held = Position(position.north[lines], position.east[across], True, grid)
    inside = np.empty(held.sizes, dtype=bool)
    for block in geometry.line_blocks(held.sizes):
        north, east = held.crossed_part(block)
        radius = np.hypot(north, east)
        turned = np.degrees(np.arctan2(east, sign * north)) + grid[0]
        inside[block] = (
            (radius >= band[0])
            & (radius <= band[1])
            & ((turned - frame.west_lon) % TURN <= span)
        )
    if not inside.any():
        return None
    return (lines, across), geometry.partial_mask(inside)


def placed(position: Position, frame: Tile) -> Position:
    """Return where the samples one cut keeps sit, in the metres of the tile's pole.

    Args:
        position: The samples the cut keeps, on the grid they were placed on.
        frame: The tile's local frame at a pole, whose centre the offsets stand from.

    Returns:
        position: The metres north and east of that centre.
    """
    place = frame.grid
    centre_x, centre_y = map(
        float, geodesy.stereographic_forward(frame.centre_lon, frame.centre_lat, *place)
    )
    grid = position.grid
    # A grid of the tile's own pole and centre longitude reaches it by a scale.
    if grid is not None and grid[:2] == place[:2]:
        scale = place[2] / grid[2]
        return Position(
            position.north * scale - centre_y,
            position.east * scale - centre_x,
            position.separable,
            place,
        )
    north, east = geometry.filled_offsets(
        position, partial(projected_offsets, position, place, (centre_x, centre_y))
    )
    return Position(north, east, False, place)


def projected_offsets(
    position: Position, place: PolarGrid, centre: tuple[float, float], block: tuple
) -> tuple[np.ndarray, np.ndarray]:
    """Return the metres one block of samples stands off, projected onto the pole.

    Args:
        position: The samples the cut keeps, on the grid they were placed on.
        place: The grid the tile is read on.
        centre: The tile centre's easting and northing on that grid.
        block: Which lines of the cut to read.

    Returns:
        north: Their metres north of the centre.
        east: Their metres east of it.
    """
    lon, lat = geometry.block_degrees(position, block)
    x, y = geodesy.stereographic_forward(lon, lat, *place)
    return y - centre[1], x - centre[0]
