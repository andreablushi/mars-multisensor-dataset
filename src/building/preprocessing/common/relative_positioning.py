"""Where the samples of one position sit in degrees, and the ground each one spans."""

from __future__ import annotations

import numpy as np

from building.preprocessing.common.models.position import Position
from common.maths import geodesy
from common.models.tile import Tile

# How many samples from the middle of one axis to measure a ground sample over.
SPACING_SAMPLES = 512


def position_degrees(
    position: Position, frame: Tile, taken: tuple
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
    north, east = position.crossed_part(taken)
    if position.grid is None:
        return (
            geodesy.normalise_longitude(frame.centre_lon + east),
            frame.centre_lat + north,
        )
    # The offsets stand from the tile centre, so where that falls is worked again.
    centre_x, centre_y = geodesy.stereographic_forward(
        frame.centre_lon, frame.centre_lat, *position.grid
    )
    return geodesy.stereographic_inverse(
        east + centre_x, north + centre_y, *position.grid
    )


def sample_spacing_m(position: Position, frame: Tile) -> tuple[float, ...]:
    """Return how much ground one sample spans, along each of its ground axes.

    Args:
        position: Where the samples sit, in degrees or projected metres.
        frame: The tile's local frame, which the offsets are relative to.

    Returns:
        spacing: The median geodesic metres between neighbours per ground axis.
    """
    sizes = position.sizes
    steps: list[float] = []
    for axis, length in enumerate(sizes):
        kept = min(length, SPACING_SAMPLES)
        start = (length - kept) // 2
        # Only one line is crossed, so a projected grid is never held whole here.
        taken = tuple(
            slice(start, start + kept)
            if other == axis
            else slice(size // 2, size // 2 + 1)
            for other, size in enumerate(sizes)
        )
        lon, lat = np.broadcast_arrays(*position_degrees(position, frame, taken))
        # Each pair is walked on the spheroid itself, not on one sphere for all.
        walk = geodesy.geodesic_steps(np.ravel(lon), np.ravel(lat))
        steps.append(float(np.median(walk)) if walk.size else float("nan"))
    return tuple(steps)
