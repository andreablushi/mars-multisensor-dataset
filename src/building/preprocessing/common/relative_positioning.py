"""Turning the coordinates an observation carries into offsets from its feature."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from building.metadata.models.feature import FeatureFrame
from building.preprocessing.common.models.relative_position import RelativePosition
from utils.geometry import geodesy

# How many neighbouring pairs of one axis to measure a ground sample over.
MEASURED = 512


class Positioned(Protocol):
    """What every instrument's observation says about where its own samples sit.

    Attributes:
        latitude: The latitude of every sample, or of every line where the grid
            is separable.
        longitude: The longitude of every sample, or of every sample of a line.
        separable: Whether those two hold one axis each rather than a value for
            every sample.
    """

    latitude: np.ndarray
    longitude: np.ndarray
    separable: bool


def relative_position(observation: Positioned, frame: FeatureFrame) -> RelativePosition:
    """Return where every sample of one observation sits on its feature.

    Args:
        observation: The observation as it was read off disk, saying where its
            own samples were measured.
        frame: The local frame of the feature it was kept for.

    Returns:
        The position, in degrees from that feature's own centre.
    """
    return RelativePosition(
        north=observation.latitude - frame.centre_lat,
        east=geodesy.normalise_longitude(observation.longitude - frame.centre_lon),
        separable=observation.separable,
    )


def ground_sample_m(
    position: RelativePosition, frame: FeatureFrame
) -> tuple[float, ...]:
    """Return how much ground one sample spans, along each of its ground axes.

    Args:
        position: Where the samples sit, in degrees from the feature centre.
        frame: The feature's local frame, which the offsets are relative to.

    Returns:
        The median great-circle distance in metres between samples neighbouring
        along each ground axis, in the order those axes run, and not a number
        for an axis holding a single sample. A map raster near a pole is far
        finer across than along, so one figure for both would say neither.
    """
    lat = frame.centre_lat + position.north
    lon = frame.centre_lon + position.east

    def middle(length: int) -> slice:
        """Return at most MEASURED samples from the middle of one axis.

        Args:
            length: How many samples the axis holds.

        Returns:
            The slice of it to measure over.
        """
        kept = min(length, MEASURED)
        start = (length - kept) // 2
        return slice(start, start + kept)

    steps: list[float] = []
    for axis in range(position.ground_axes):
        if position.separable:
            # One axis holds every line's latitude and the other every sample's
            # longitude, so the walked one is read across the middle of the other.
            if axis == 0:
                walked = lat[middle(lat.size)]
                line = (np.full(walked.size, lon[lon.size // 2]), walked)
            else:
                walked = lon[middle(lon.size)]
                line = (walked, np.full(walked.size, lat[lat.size // 2]))
        else:
            taken = tuple(
                middle(size) if held == axis else slice(size // 2, size // 2 + 1)
                for held, size in enumerate(lat.shape)
            )
            line = (np.ravel(lon[taken]), np.ravel(lat[taken]))
        walk = geodesy.haversine_steps(*line)
        steps.append(float(np.median(walk)) if walk.size else float("nan"))
    return tuple(steps)
