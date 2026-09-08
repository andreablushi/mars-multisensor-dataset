"""Turning the coordinates an observation carries into offsets from its feature."""

from __future__ import annotations

import numpy as np

from building.preprocessing.common.models.relative_position import RelativePosition
from shared.maths import geodesy
from shared.models.feature import Feature

# How many neighbouring pairs of one axis to measure a ground sample over.
MEASURED = 512


def degrees(
    position: RelativePosition, frame: Feature, taken: tuple = ()
) -> tuple[np.ndarray, np.ndarray]:
    """Return the longitude and latitude the samples of one position sit at.

    Args:
        position: Where the samples sit, in degrees from the feature centre or
            in the metres of the projection it was placed on.
        frame: The feature's local frame, which those offsets are relative to.
        taken: Which of each ground axis to read, outermost first, and empty for
            all of them.

    Returns:
        longitudes: The longitudes in degrees.
        latitudes: The latitudes in degrees.
    """
    down, across = position.offsets(taken)
    if position.polar is None:
        return (
            geodesy.normalise_longitude(frame.centre_lon + across),
            frame.centre_lat + down,
        )
    # The offsets stand from the feature centre, so where that falls is worked again.
    centre_x, centre_y = geodesy.stereographic_forward(
        frame.centre_lon, frame.centre_lat, *position.polar
    )
    x, y = across + centre_x, down + centre_y
    if position.separable:
        x, y = x[None, :], y[:, None]
    return geodesy.stereographic_inverse(x, y, *position.polar)


def ground_sample_m(position: RelativePosition, frame: Feature) -> tuple[float, ...]:
    """Return how much ground one sample spans, along each of its ground axes.

    Args:
        position: Where the samples sit, in degrees from the feature centre or
            in the metres of the projection it was placed on.
        frame: The feature's local frame, which the offsets are relative to.

    Returns:
        sample: The median great-circle metres between neighbouring samples along each
            ground axis, and not a number for an axis holding a single sample.
    """

    def middle(length: int) -> slice:
        """Return at most MEASURED samples from the middle of one axis.

        Args:
            length: How many samples the axis holds.

        Returns:
            middle: The slice of it to measure over.
        """
        kept = min(length, MEASURED)
        start = (length - kept) // 2
        return slice(start, start + kept)

    plain = position.separable and position.polar is None
    sizes = position.ground_sizes
    held = degrees(position, frame) if plain else None
    steps: list[float] = []
    for axis in range(len(sizes)):
        if plain:
            # One axis holds latitude, the other longitude, walked at the middle.
            lon, lat = held
            if axis == 0:
                walked = lat[middle(lat.size)]
                line = (np.full(walked.size, lon[lon.size // 2]), walked)
            else:
                walked = lon[middle(lon.size)]
                line = (walked, np.full(walked.size, lat[lat.size // 2]))
        else:
            # Only one line is crossed, so a projected grid is never held whole here.
            taken = tuple(
                middle(size) if held == axis else slice(size // 2, size // 2 + 1)
                for held, size in enumerate(sizes)
            )
            lon, lat = degrees(position, frame, taken)
            line = (np.ravel(lon), np.ravel(lat))
        # The spheroid is measured where each pair stands, not on one sphere for all.
        middles = (line[1][:-1] + line[1][1:]) / 2.0
        walk = geodesy.haversine_steps(*line, geodesy.local_radius_m(middles))
        steps.append(float(np.median(walk)) if walk.size else float("nan"))
    return tuple(steps)
