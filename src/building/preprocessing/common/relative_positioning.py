"""Turning the coordinates an observation carries into offsets from its tile."""

from __future__ import annotations

import numpy as np

from building.preprocessing.common import geometry
from building.preprocessing.common.models.relative_position import (
    RelativePosition,
)
from common.maths import geodesy
from common.models.tile import Tile

# How many neighbouring pairs of one axis to measure a ground sample over.
MEASURED = 512

# How many samples of a crop become metres at once, since a scan can be huge.
BLOCK = 1_000_000

# What the offsets are stored as, which holds a centimetre over any tile.
STORED = np.float32


def position_degrees(
    position: RelativePosition, frame: Tile, taken: tuple
) -> tuple[np.ndarray, np.ndarray]:
    """Return the longitude and latitude the samples of one position sit at.

    Args:
        position: Where the samples sit, in degrees or projected metres.
        frame: The tile's local frame, which those offsets are relative to.
        taken: Which of each ground axis to read, outermost first.

    Returns:
        longitudes: The longitudes in degrees, crossed where the axes are separable.
        latitudes: The latitudes in degrees, holding the same.
    """
    down, across = position.offsets(taken)
    if position.separable:
        down, across = down[:, None], across[None, :]
    if position.polar is None:
        return (
            geodesy.normalise_longitude(frame.centre_lon + across),
            frame.centre_lat + down,
        )
    # The offsets stand from the tile centre, so where that falls is worked again.
    centre_x, centre_y = geodesy.stereographic_forward(
        frame.centre_lon, frame.centre_lat, *position.polar
    )
    return geodesy.stereographic_inverse(
        across + centre_x, down + centre_y, *position.polar
    )


def distance_centre_m(
    position: RelativePosition, frame: Tile
) -> tuple[np.ndarray, np.ndarray]:
    """Return how far north and east of its tile centre every sample sits.

    Args:
        position: Where the samples sit, in degrees or projected metres.
        frame: The tile's local frame, which the offsets are measured from.

    Returns:
        north: The ground metres north of that centre, one per sample.
        east: The ground metres east of it, in the same frame.
    """
    sizes = position.ground_sizes
    north = np.empty(sizes, dtype=STORED)
    east = np.empty(sizes, dtype=STORED)
    for block in geometry.line_blocks(sizes, BLOCK):
        taken = (block, *(slice(None),) * (len(sizes) - 1))
        lon, lat = position_degrees(position, frame, taken)
        east[block], north[block] = geodesy.geodesic_forward(
            lon, lat, frame.centre_lon, frame.centre_lat
        )
    return north, east


def middle_slice(length: int) -> slice:
    """Return at most MEASURED samples from the middle of one axis.

    Args:
        length: How many samples the axis holds.

    Returns:
        middle: The slice of it to measure over.
    """
    kept = min(length, MEASURED)
    start = (length - kept) // 2
    return slice(start, start + kept)


def sample_spacing_m(position: RelativePosition, frame: Tile) -> tuple[float, ...]:
    """Return how much ground one sample spans, along each of its ground axes.

    Args:
        position: Where the samples sit, in degrees or projected metres.
        frame: The tile's local frame, which the offsets are relative to.

    Returns:
        spacing: The median geodesic metres between neighbours per ground axis.
    """
    sizes = position.ground_sizes
    steps: list[float] = []
    for axis in range(len(sizes)):
        # Only one line is crossed, so a projected grid is never held whole here.
        taken = tuple(
            middle_slice(size) if other == axis else slice(size // 2, size // 2 + 1)
            for other, size in enumerate(sizes)
        )
        lon, lat = np.broadcast_arrays(*position_degrees(position, frame, taken))
        # Each pair is walked on the spheroid itself, not on one sphere for all.
        walk = geodesy.geodesic_steps(np.ravel(lon), np.ravel(lat))
        steps.append(float(np.median(walk)) if walk.size else float("nan"))
    return tuple(steps)
