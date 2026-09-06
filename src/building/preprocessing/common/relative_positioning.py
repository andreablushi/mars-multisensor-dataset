"""Turning the coordinates an observation carries into offsets from its feature."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from building.models.feature import FeatureFrame
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


def degrees(
    position: RelativePosition, frame: FeatureFrame, taken: tuple = ()
) -> tuple[np.ndarray, np.ndarray]:
    """Return the longitude and latitude the samples of one position sit at.

    Args:
        position: Where the samples sit, in degrees from the feature centre or
            in the metres of the projection it was placed on.
        frame: The feature's local frame, which those offsets are relative to.
        taken: Which of each ground axis to read, outermost first, empty for
            all of them. A projected grid is crossed to be read, so this keeps
            the crossing to the part that is wanted.

    Returns:
        The longitudes and latitudes in degrees. They hold one axis each where
        the position is separable degrees, and are crossed over both axes where
        it is separable metres on a projection.
    """
    if position.separable:
        down = position.north[taken[0]] if taken else position.north
        across = position.east[taken[1]] if taken else position.east
    else:
        down = position.north[taken] if taken else position.north
        across = position.east[taken] if taken else position.east
    if position.polar is None:
        return frame.centre_lon + across, frame.centre_lat + down
    # The offsets stand from the feature centre, so where that falls is worked again.
    centre_x, centre_y = geodesy.stereographic_forward(
        frame.centre_lon, frame.centre_lat, *position.polar
    )
    x, y = across + centre_x, down + centre_y
    if position.separable:
        x, y = x[None, :], y[:, None]
    return geodesy.stereographic_inverse(x, y, *position.polar)


def metres(
    position: RelativePosition, frame: FeatureFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Return where every sample sits, in metres north and east of its feature.

    Args:
        position: Where the samples sit, in degrees from the feature centre or
            in the metres of the projection it was placed on.
        frame: The feature's local frame, which those offsets are relative to.

    Returns:
        The northing and the easting in metres. A separable position in degrees
        keeps its northing on the one axis it was held over and spreads its
        easting over both, since a degree of longitude covers less ground the
        further north it is measured; every other position is crossed already.
    """
    if position.polar is not None:
        # A projection's metres are its own, so the ground comes off its degrees.
        lon, lat = degrees(position, frame)
        radius = geodesy.local_radius_m(lat)
        north = np.radians(lat - frame.centre_lat) * radius
        east = np.radians(geodesy.normalise_longitude(lon - frame.centre_lon))
        return north, east * radius * np.cos(np.radians(lat))
    lat = frame.centre_lat + position.north
    radius = geodesy.local_radius_m(lat)
    reach = radius * np.cos(np.radians(lat))
    north = np.radians(position.north) * radius
    east = np.radians(position.east)
    return north, east * (reach[:, None] if position.separable else reach)


def ground_sample_m(
    position: RelativePosition, frame: FeatureFrame
) -> tuple[float, ...]:
    """Return how much ground one sample spans, along each of its ground axes.

    Args:
        position: Where the samples sit, in degrees from the feature centre or
            in the metres of the projection it was placed on.
        frame: The feature's local frame, which the offsets are relative to.

    Returns:
        The median great-circle distance in metres between samples neighbouring
        along each ground axis, in the order those axes run, and not a number
        for an axis holding a single sample. A map raster near a pole is far
        finer across than along, so one figure for both would say neither, and
        a projection's own metres are not the ground's, so both are measured
        off the degrees rather than read from the offsets.
    """

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

    plain = position.separable and position.polar is None
    sizes = (
        (position.north.size, position.east.size)
        if position.separable
        else position.north.shape
    )
    steps: list[float] = []
    for axis in range(position.ground_axes):
        if plain:
            # One axis holds latitude, the other longitude, walked at the middle.
            lon, lat = degrees(position, frame)
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
