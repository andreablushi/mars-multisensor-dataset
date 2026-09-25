"""The projection a tile at a pole is read on: stereographic metres from its centre."""

from __future__ import annotations

import math
from functools import partial

import numpy as np

from building.preprocessing.common import geometry
from building.preprocessing.common.models.relative_position import (
    RelativePosition,
)
from building.preprocessing.common.models.samples import Samples
from common.maths import geodesy
from common.maths.geodesy import TURN, PolarGrid
from common.models.tile import Tile

# The longest segment the box is walked in, a chord leaving its arc by under a pixel.
STEP = 0.1


def cut(
    samples: Samples, frame: Tile, span: float
) -> tuple[tuple[np.ndarray, ...], np.ndarray | None] | None:
    """Return which bins of one grid projected onto a pole the box keeps.

    Args:
        samples: The samples, placed in the metres of the grid they sit on.
        frame: The tile's local frame, carrying the box the catalogue gives it.
        span: How many degrees of longitude that box covers.

    Returns:
        bounds: The bins to keep of each ground axis, outermost first.
        inside: Which kept bins truly fall in the box, or None for all.
        None: Where the grid reaches none of the box.
    """
    grid = samples.grid
    ring_x, ring_y = geodesy.stereographic_forward(
        *geodesy.bbox_ring(
            frame.min_lat, frame.max_lat, frame.west_lon, frame.east_lon, STEP
        ),
        *grid,
    )
    # The box projects to a sector, and the ring its edge traces bounds it.
    lines = np.flatnonzero(
        (samples.down >= ring_y.min()) & (samples.down <= ring_y.max())
    )
    across = np.flatnonzero(
        (samples.across >= ring_x.min()) & (samples.across <= ring_x.max())
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
    held = Samples(samples.down[lines], samples.across[across], True, grid)
    inside = np.empty(held.sizes, dtype=bool)
    for block in geometry.line_blocks(held.sizes):
        down, east = geometry.block_axes(held, block)
        radius = np.hypot(down, east)
        turned = np.degrees(np.arctan2(east, sign * down)) + grid[0]
        inside[block] = (
            (radius >= band[0])
            & (radius <= band[1])
            & ((turned - frame.west_lon) % TURN <= span)
        )
    if not inside.any():
        return None
    return (lines, across), geometry.partial_mask(inside)


def placed(samples: Samples, frame: Tile) -> RelativePosition:
    """Return where the samples one cut keeps sit, in the metres of the tile's pole.

    Args:
        samples: The samples the cut keeps, on the grid they were placed on.
        frame: The tile's local frame at a pole, whose centre the offsets stand from.

    Returns:
        position: The metres north and east of that centre.
    """
    place = frame.grid
    centre_x, centre_y = map(
        float, geodesy.stereographic_forward(frame.centre_lon, frame.centre_lat, *place)
    )
    grid = samples.grid
    # A grid of the tile's own pole reaches it by a turn and a scale.
    if grid is not None and grid[1] == place[1]:
        scale = place[2] / grid[2]
        turned = math.radians(grid[0] - place[0])
        if not turned:
            return RelativePosition(
                samples.down * scale - centre_y,
                samples.across * scale - centre_x,
                samples.separable,
                place,
            )
        offsets = partial(
            turned_offsets, samples, place, (centre_x, centre_y), scale, turned
        )
    else:
        offsets = partial(projected_offsets, samples, place, (centre_x, centre_y))
    north, east = geometry.filled_offsets(samples, offsets)
    return RelativePosition(north, east, False, place)


def turned_offsets(
    samples: Samples,
    place: PolarGrid,
    centre: tuple[float, float],
    scale: float,
    turned: float,
    block: slice,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the metres one block of samples on a grid of the same pole stands off.

    Args:
        samples: The samples the cut keeps, on the grid they were placed on.
        place: The grid the tile is read on.
        centre: The tile centre's easting and northing on that grid.
        scale: How many of the tile grid's metres one of the samples' grid is.
        turned: How far the samples' grid is turned from the tile's, in radians.
        block: Which lines of the cut to read.

    Returns:
        north: Their metres north of the centre.
        east: Their metres east of it.
    """
    down, across = geometry.block_axes(samples, block)
    # A pole's eastings run the other way round, so its turn does too.
    sign = 1.0 if place[1] else -1.0
    cosine, sine = math.cos(turned), math.sin(turned)
    return (
        scale * (down * cosine + sign * across * sine) - centre[1],
        scale * (across * cosine - sign * down * sine) - centre[0],
    )


def projected_offsets(
    samples: Samples, place: PolarGrid, centre: tuple[float, float], block: slice
) -> tuple[np.ndarray, np.ndarray]:
    """Return the metres one block of samples stands off, projected onto the pole.

    Args:
        samples: The samples the cut keeps, on the grid they were placed on.
        place: The grid the tile is read on.
        centre: The tile centre's easting and northing on that grid.
        block: Which lines of the cut to read.

    Returns:
        north: Their metres north of the centre.
        east: Their metres east of it.
    """
    lon, lat = geometry.block_degrees(samples, block)
    x, y = geodesy.stereographic_forward(lon, lat, *place)
    return y - centre[1], x - centre[0]
