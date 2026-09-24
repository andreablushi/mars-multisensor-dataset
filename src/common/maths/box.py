"""Boxes on Mars bounded by latitudes and longitudes: how two meet, and map crops."""

from __future__ import annotations

from dataclasses import replace
from typing import NamedTuple

import numpy as np

from common.maths.geodesy import TURN, longitude_span, longitude_stretch

POLE = 90.0

# A box as its southern and northern latitudes, its west edge and its eastward span.
Box = tuple[
    np.ndarray | float, np.ndarray | float, np.ndarray | float, np.ndarray | float
]


def bounds_box(bounded) -> Box:
    """Return the box one tile or one feature is bounded by.

    Args:
        bounded: Anything bounded by two latitudes and two longitudes.

    Returns:
        box: Its latitudes, its west edge and its eastward span.
    """
    span = longitude_span(bounded.west_lon, bounded.east_lon)
    return bounded.min_lat, bounded.max_lat, bounded.west_lon, span


def bounds_boxes(bounded) -> Box:
    """Return the boxes many tiles or features are bounded by, stacked.

    Args:
        bounded: Everything to stack, each bounded by two latitudes and two longitudes.

    Returns:
        boxes: Their latitudes, west edges and eastward spans, one array each.
    """
    return tuple(
        np.array(edges) for edges in zip(*map(bounds_box, bounded), strict=True)
    )


def recut[Bounded](bounded: Bounded, cut) -> Bounded:
    """Return a copy of one bounded record, bounded by another's box instead.

    Args:
        bounded: The dataclass to copy.
        cut: Anything bounded by two latitudes and two longitudes.

    Returns:
        recut: The copy, holding the box of the cut.
    """
    return replace(
        bounded,
        min_lat=cut.min_lat,
        max_lat=cut.max_lat,
        west_lon=cut.west_lon,
        east_lon=cut.east_lon,
    )


def centre_offset(inner: Box, outer: Box) -> np.ndarray:
    """Return how far each inner box's centre sits from its outer box's centre.

    Args:
        inner: The boxes to measure, one or an array of them.
        outer: The boxes they are measured in, one or an array of them.

    Returns:
        offset: One share per pair, 0 at the centre and 1 at the edge.
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


class Crop(NamedTuple):
    """One plate carree lon/lat box, as a mosaic crop is asked for and drawn over.

    Attributes:
        west: Its western edge in degrees.
        south: Its southern edge in degrees.
        east: Its eastern edge in degrees.
        north: Its northern edge in degrees.
    """

    west: float
    south: float
    east: float
    north: float

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """Return the crop as an image extent."""
        return self.west, self.east, self.south, self.north

    @property
    def centre_lat(self) -> float:
        """Return the latitude the crop is centred on."""
        return (self.south + self.north) / 2.0


def crop_around(lon: np.ndarray, lat: np.ndarray, min_span_deg: float) -> Crop:
    """Return the crop holding every point, each side held open to a minimum.

    Args:
        lon: The longitudes to hold, on one turn.
        lat: The latitudes to hold.
        min_span_deg: The least ground each side spans, in degrees of latitude.

    Returns:
        crop: The crop, widened about its middle where a side falls short.
    """
    centre_lat = float((lat.min() + lat.max()) / 2.0)
    south, north = floored(float(lat.min()), float(lat.max()), min_span_deg)
    west, east = floored(
        float(lon.min()),
        float(lon.max()),
        min_span_deg / longitude_stretch(centre_lat),
    )
    return Crop(west, south, east, north)


def floored(low: float, high: float, minimum: float) -> tuple[float, float]:
    """Hold one side of a box open to a minimum width, about its middle.

    Args:
        low: The lower edge.
        high: The upper edge.
        minimum: The width to hold it open to.

    Returns:
        edges: The edges, widened about their middle when closer than the minimum.
    """
    if high - low >= minimum:
        return low, high
    centre = (low + high) / 2.0
    return centre - minimum / 2.0, centre + minimum / 2.0
