"""Boxes on Mars bounded by latitudes and longitudes, and how two of them meet."""

from __future__ import annotations

import numpy as np

from analysis.ground_truth.models.feature import Feature
from common.maths.geodesy import TURN, longitude_span

POLE = 90.0

# A box as its southern and northern latitudes, its west edge and its eastward span.
Box = tuple[
    np.ndarray | float, np.ndarray | float, np.ndarray | float, np.ndarray | float
]


def box_span(box) -> float:
    """Return how far east a box reaches from its west edge.

    Args:
        box: Anything bounded by a west and an east longitude.

    Returns:
        span: The eastward span in degrees, 360 where it circles a pole.
    """
    return longitude_span(box.west_lon, box.east_lon)


def bounds_box(bounded) -> Box:
    """Return the box one tile or one feature is bounded by.

    Args:
        bounded: Anything bounded by two latitudes and two longitudes.

    Returns:
        box: Its latitudes, its west edge and its eastward span.
    """
    return bounded.min_lat, bounded.max_lat, bounded.west_lon, box_span(bounded)


def claimed_box(feature: Feature, latitudes: tuple[float, float] | None) -> Box | None:
    """Return the part of a feature's box a texture tile has to lie in.

    Args:
        feature: The feature.
        latitudes: The latitudes the class is kept to, or None for anywhere.

    Returns:
        box: The box, or None where the latitudes leave none of it.
    """
    south, north, west, span = bounds_box(feature)
    if latitudes is not None:
        south, north = max(south, latitudes[0]), min(north, latitudes[1])
    return (south, north, west, span) if south < north else None


def centre_offset(inner: Box, outer: Box) -> np.ndarray:
    """Return how far each inner box's centre sits from its outer box's centre.

    Args:
        inner: The boxes to measure, one or an array of them.
        outer: The boxes they are measured in, one or an array of them.

    Returns:
        offset: One share per pair, 0 at the centre and 1 at the edge, which is
            measured from the pole for a box circling one.
    """
    south, north, west, span = outer
    latitude = (np.asarray(inner[0]) + inner[1]) / 2.0
    height = np.asarray(north) - south
    pole = np.where(np.asarray(north) >= POLE, north, south)
    eastward = (np.asarray(inner[2]) + np.asarray(inner[3]) / 2.0 - west) % TURN
    plain = np.maximum(
        np.abs(2.0 * latitude - south - north) / height,
        np.abs(2.0 * eastward - span) / span,
    )
    return np.where(span >= TURN, np.abs(pole - latitude) / height, plain)


def inside(inner: Box, outer: Box) -> np.ndarray:
    """Return whether each inner box lies whole in its outer box.

    Args:
        inner: The boxes to test, one or an array of them.
        outer: The boxes they have to lie in, one or an array of them.

    Returns:
        inside: One flag per pair.
    """
    offset = (np.asarray(inner[2]) - outer[2]) % TURN
    return (
        (np.asarray(inner[0]) >= outer[0])
        & (np.asarray(inner[1]) <= outer[1])
        & ((np.asarray(outer[3]) >= TURN) | (offset + inner[3] <= outer[3]))
    )


def touching(one: Box, other: Box) -> np.ndarray:
    """Return whether each pair of boxes shares any ground.

    Args:
        one: The boxes to test, one or an array of them.
        other: The boxes they are tested against, one or an array of them.

    Returns:
        touching: One flag per pair.
    """
    return (
        (np.asarray(one[0]) < other[1])
        & (np.asarray(one[1]) > other[0])
        & (
            ((np.asarray(one[2]) - other[2]) % TURN < other[3])
            | ((np.asarray(other[2]) - one[2]) % TURN < one[3])
        )
    )
