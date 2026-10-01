"""Geometry on Mars: longitudes, projection, areas, and geodesic lengths."""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from pyproj import Geod

from common.maths.physics import EQUATORIAL_RADIUS_M, POLAR_RADIUS_M, RADIUS_M

# The whole turn, which every longitude here is measured round.
TURN = 360.0

HALF_TURN = 180.0

POLE = 90.0

# A degree of longitude vanishes at a pole, so the correction is floored
MIN_COSINE = 0.05

# How near the antipode the projection is allowed to divide by
LAEA_MIN_DENOMINATOR = 1e-12

# The spheroid every ground distance is walked on, which no sphere stands in for.
SPHEROID = Geod(a=EQUATORIAL_RADIUS_M, b=POLAR_RADIUS_M)


class PolarGrid(NamedTuple):
    """One polar stereographic grid, by its centre, its pole and its sphere.

    Attributes:
        centre_lon: The longitude the projection is centred on, in degrees.
        north: Whether it is centred on the north pole rather than the south.
        radius_m: The sphere the projection is built on, in metres.
    """

    centre_lon: float
    north: bool
    radius_m: float


def normalise_longitude(lon: np.ndarray | float) -> np.ndarray:
    """Wrap one longitude or an array of them into -180 to 180 degrees, as floats."""
    return (np.asarray(lon, dtype=float) + HALF_TURN) % TURN - HALF_TURN


def longitude_stretch(lat: float) -> float:
    """Return how a degree of longitude shrinks at a latitude, at least MIN_COSINE."""
    return max(math.cos(math.radians(lat)), MIN_COSINE)


def longitude_span(west_lon: float, east_lon: float) -> float:
    """Return the eastward span in degrees from a west to an east longitude.

    Args:
        west_lon: The westernmost longitude in degrees.
        east_lon: The easternmost longitude in degrees.

    Returns:
        span: The eastward span in degrees, above zero and up to 360.
    """
    raw = east_lon - west_lon
    if raw >= TURN or raw == 0.0:
        return TURN
    return raw % TURN


def bbox_centre(bounded) -> tuple[float, float]:
    """Return the centre of the box one tile or one feature is bounded by.

    Args:
        bounded: Anything bounded by two latitudes and two longitudes.

    Returns:
        longitude: The centre longitude in -180 to 180 degrees.
        latitude: The centre latitude in degrees.
    """
    west = bounded.west_lon
    centre_lon = west + longitude_span(west, bounded.east_lon) / 2.0
    centre_lat = (bounded.min_lat + bounded.max_lat) / 2.0
    return float(normalise_longitude(centre_lon)), centre_lat


def bbox_ring(bounded, step: float) -> tuple[np.ndarray, np.ndarray]:
    """Return a densified closed lon/lat ring tracing the box one tile is bounded by.

    Args:
        bounded: Anything bounded by two latitudes and two longitudes.
        step: The longest segment the ring is densified to, in degrees.

    Returns:
        longitudes: The ring longitudes, closed back onto the first point.
        latitudes: The ring latitudes, closed the same way.
    """
    south, north, west = bounded.min_lat, bounded.max_lat, bounded.west_lon
    span = longitude_span(west, bounded.east_lon)
    east = west + span
    along = np.linspace(west, east, max(2, math.ceil(span / step) + 1))
    up = np.linspace(south, north, max(2, math.ceil((north - south) / step) + 1))
    lons = np.concatenate(
        [along, np.full(up.size, east), along[::-1], np.full(up.size, west)]
    )
    lats = np.concatenate(
        [np.full(along.size, south), up, np.full(along.size, north), up[::-1]]
    )
    return lons, lats


def laea_forward(
    lon: np.ndarray | float,
    lat: np.ndarray | float,
    centre_lon: float,
    centre_lat: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project lon/lat degrees into Lambert azimuthal equal-area metres.

    Args:
        lon: The longitudes in degrees.
        lat: The latitudes in degrees.
        centre_lon: The projection centre longitude in degrees.
        centre_lat: The projection centre latitude in degrees.

    Returns:
        eastings: The projected eastings in metres.
        northings: The projected northings in metres.
    """
    delta = np.radians(np.asarray(lon, dtype=float) - centre_lon)
    phi = np.radians(np.asarray(lat, dtype=float))
    phi0 = math.radians(centre_lat)
    cos_c = math.sin(phi0) * np.sin(phi) + math.cos(phi0) * np.cos(phi) * np.cos(delta)
    scale = np.sqrt(2.0 / np.maximum(1.0 + cos_c, LAEA_MIN_DENOMINATOR))
    x = RADIUS_M * scale * np.cos(phi) * np.sin(delta)
    y = (
        RADIUS_M
        * scale
        * (math.cos(phi0) * np.sin(phi) - math.sin(phi0) * np.cos(phi) * np.cos(delta))
    )
    return x, y


def geodesic_steps(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Return the geodesic distance between each neighbouring pair of points.

    Args:
        lon: The point longitudes in degrees.
        lat: The point latitudes in degrees.

    Returns:
        steps: One distance in metres per neighbouring pair.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    _, _, steps = SPHEROID.inv(lon[:-1], lat[:-1], lon[1:], lat[1:])
    return steps


def northward_m(degrees: float) -> float:
    """Return how far north a span of latitude in degrees reaches, in metres."""
    return math.radians(degrees) * RADIUS_M


def spheroid_radius_m(lat: float) -> float:
    """Return how far the spheroid's surface stands from the centre at one latitude.

    Args:
        lat: The planetocentric latitude in degrees.

    Returns:
        radius: The distance in metres.
    """
    held = math.radians(lat)
    across = POLAR_RADIUS_M * math.cos(held)
    up = EQUATORIAL_RADIUS_M * math.sin(held)
    return EQUATORIAL_RADIUS_M * POLAR_RADIUS_M / math.hypot(across, up)


def laea_inverse(
    x: np.ndarray | float,
    y: np.ndarray | float,
    centre_lon: float,
    centre_lat: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn Lambert azimuthal equal-area metres back into lon/lat degrees.

    Args:
        x: The projected eastings in metres.
        y: The projected northings in metres.
        centre_lon: The projection centre longitude in degrees.
        centre_lat: The projection centre latitude in degrees.

    Returns:
        longitudes: The longitudes in degrees, wrapped to -180 to 180.
        latitudes: The latitudes in degrees.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    phi0 = math.radians(centre_lat)
    rho = np.hypot(x, y)
    safe = np.where(rho == 0.0, 1.0, rho)
    c = 2.0 * np.arcsin(np.clip(rho / (2.0 * RADIUS_M), -1.0, 1.0))
    sin_c, cos_c = np.sin(c), np.cos(c)
    lat = np.arcsin(
        np.clip(cos_c * math.sin(phi0) + y * sin_c * math.cos(phi0) / safe, -1.0, 1.0)
    )
    lon = np.radians(centre_lon) + np.arctan2(
        x * sin_c,
        safe * math.cos(phi0) * cos_c - y * math.sin(phi0) * sin_c,
    )
    return normalise_longitude(np.degrees(lon)), np.degrees(
        np.where(rho == 0.0, phi0, lat)
    )


def stereographic_forward(
    lon: np.ndarray | float,
    lat: np.ndarray | float,
    centre_lon: float,
    north: bool,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project lon/lat degrees into polar stereographic metres.

    Args:
        lon: The longitudes in degrees.
        lat: The latitudes in degrees.
        centre_lon: The longitude the projection is centred on, in degrees.
        north: Whether it is centred on the north pole rather than the south.
        radius: The sphere the projection is built on, in metres.

    Returns:
        eastings: The projected eastings in metres.
        northings: The projected northings in metres.
    """
    lam = np.radians(np.asarray(lon, dtype=float) - centre_lon)
    phi = np.radians(np.asarray(lat, dtype=float))
    quarter = math.pi / 4.0
    rho = 2.0 * radius * np.tan(quarter - (phi if north else -phi) / 2.0)
    return rho * np.sin(lam), (-1.0 if north else 1.0) * rho * np.cos(lam)


def stereographic_inverse(
    x: np.ndarray | float,
    y: np.ndarray | float,
    centre_lon: float,
    north: bool,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn polar stereographic metres back into lon/lat degrees.

    Args:
        x: The projected eastings in metres.
        y: The projected northings in metres.
        centre_lon: The longitude the projection is centred on, in degrees.
        north: Whether it is centred on the north pole rather than the south.
        radius: The sphere the projection is built on, in metres.

    Returns:
        longitudes: The longitudes in -180 to 180 degrees.
        latitudes: The latitudes in degrees.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    angle = 2.0 * np.arctan2(np.hypot(x, y), 2.0 * radius)
    lat = np.degrees(math.pi / 2.0 - angle) * (1.0 if north else -1.0)
    lon = centre_lon + np.degrees(np.arctan2(x, -y if north else y))
    return normalise_longitude(lon), lat
