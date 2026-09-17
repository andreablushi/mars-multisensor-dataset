"""The projection a tile at a pole is read on: stereographic metres from its centre."""

from __future__ import annotations

import math

import numpy as np

from building.preprocessing.common import cross
from building.preprocessing.common.crop import marked
from building.preprocessing.common.models.cut import Cut
from building.preprocessing.common.models.relative_position import (
    PolarGrid,
    RelativePosition,
)
from building.preprocessing.common.models.samples import Samples
from shared.maths import geodesy
from shared.maths.geodesy import TURN
from shared.models.tile import Tile

# The longest segment the box is walked in, a chord leaving its arc by under a pixel.
STEP = 0.1


def cut(samples: Samples, frame: Tile, span: float) -> Cut | None:
    """Return which bins of one grid projected onto a pole the box keeps.

    Args:
        samples: The samples, placed in the metres of the grid they sit on.
        frame: The tile's local frame, carrying the box the catalogue gives it.
        span: How many degrees of longitude that box covers.

    Returns:
        held: What the box keeps, or None where the grid reaches none of it.
    """
    grid = samples.grid
    ring = geodesy.stereographic_forward(
        *geodesy.bbox_ring(
            frame.min_lat, frame.max_lat, frame.west_lon, frame.east_lon, STEP
        ),
        *grid,
    )
    # The box projects to a sector, and the ring its edge traces bounds it.
    lines = np.flatnonzero(
        (samples.down >= ring[1].min()) & (samples.down <= ring[1].max())
    )
    across = np.flatnonzero(
        (samples.across >= ring[0].min()) & (samples.across <= ring[0].max())
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
    for block in cross.blocked(held.sizes):
        down, east = cross.axes(held, block)
        radius = np.hypot(down, east)
        turned = np.degrees(np.arctan2(east, sign * down)) + grid[0]
        inside[block] = (
            (radius >= band[0])
            & (radius <= band[1])
            & ((turned - frame.west_lon) % TURN <= span)
        )
    if not inside.any():
        return None
    return Cut((lines, across), marked(inside), True)


def placed(samples: Samples, frame: Tile, place: PolarGrid) -> RelativePosition:
    """Return where the samples one cut keeps sit, in the metres of the tile's pole.

    Args:
        samples: The samples the cut keeps, on the grid they were placed on.
        frame: The tile's local frame, whose centre the offsets stand from.
        place: The grid the tile is read on, which every instrument shares.

    Returns:
        position: The metres north and east of that centre, separable only where
            the samples sit on a grid of that same pole and meridian.
    """
    centre_x, centre_y = geodesy.stereographic_forward(
        frame.centre_lon, frame.centre_lat, *place
    )
    grid = samples.grid
    # A grid of the tile's own pole reaches it by a turn and a scale, whatever
    # meridian and sphere it was published on.
    if grid is not None and grid[1] == place[1]:
        scale = place[2] / grid[2]
        turned = math.radians(grid[0] - place[0])
        if not turned:
            return RelativePosition(
                samples.down * scale - float(centre_y),
                samples.across * scale - float(centre_x),
                samples.separable,
                place,
            )

        def offsets(block: slice) -> tuple[np.ndarray, np.ndarray]:
            """Return the metres one block of samples stands from that centre.

            Args:
                block: Which lines of the cut to read.

            Returns:
                north: Their metres north of the centre.
                east: Their metres east of it.
            """
            down, across = cross.axes(samples, block)
            # A pole's eastings run the other way round, so its turn does too.
            sign = 1.0 if place[1] else -1.0
            cosine, sine = math.cos(turned), math.sin(turned)
            return (
                scale * (down * cosine + sign * across * sine) - float(centre_y),
                scale * (across * cosine - sign * down * sine) - float(centre_x),
            )
    else:

        def offsets(block: slice) -> tuple[np.ndarray, np.ndarray]:
            """Return the metres one block of samples stands from that centre.

            Args:
                block: Which lines of the cut to read.

            Returns:
                north: Their metres north of the centre.
                east: Their metres east of it.
            """
            lon, lat = cross.degrees(samples, block)
            x, y = geodesy.stereographic_forward(lon, lat, *place)
            return y - float(centre_y), x - float(centre_x)

    north, east = cross.filled(samples, offsets)
    return RelativePosition(north, east, False, place)
