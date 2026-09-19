"""Boxes on Mars bounded by latitudes and longitudes, and how two of them meet."""

from __future__ import annotations

import numpy as np

from common.maths.geodesy import TURN, longitude_span
from evaluation.analysis.models.feature import Feature

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


def feature_box(feature: Feature) -> Box:
    """Return the box ODE bounds one feature by.

    Args:
        feature: The feature.

    Returns:
        box: Its latitudes, its west edge and its eastward span.
    """
    return feature.min_lat, feature.max_lat, feature.west_lon, box_span(feature)


def core_box(
    feature: Feature, share: float, latitudes: tuple[float, float] | None
) -> Box | None:
    """Return the part of a feature's box a texture tile has to lie in.

    Args:
        feature: The feature.
        share: The share of the box kept about its centre, which is the pole for
            a feature circling one.
        latitudes: The latitudes the class is kept to, or None for anywhere.

    Returns:
        box: The core, or None where the latitudes leave none of it.
    """
    south, north, west = feature.min_lat, feature.max_lat, feature.west_lon
    span = box_span(feature)
    trimmed = (1.0 - share) * (north - south)
    if span >= TURN and abs(north) >= POLE:
        south += trimmed
    elif span >= TURN and abs(south) >= POLE:
        north -= trimmed
    else:
        south, north = south + trimmed / 2.0, north - trimmed / 2.0
        west, span = (west + (1.0 - share) * span / 2.0) % TURN, share * span
    if latitudes is not None:
        south, north = max(south, latitudes[0]), min(north, latitudes[1])
    return (south, north, west, span) if south < north else None


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
