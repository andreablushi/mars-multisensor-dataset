"""Spherical geometry on Mars: longitudes, projection, areas, and lengths."""

from __future__ import annotations

import math

import numpy as np

from shared.maths.physics import EQUATORIAL_RADIUS_M, POLAR_RADIUS_M, RADIUS_M

# The whole turn, which every longitude here is measured round.
TURN = 360.0

# A degree of longitude vanishes at a pole, so the correction is floored
MIN_COSINE = 0.05

# How near the antipode the projection is allowed to divide by
LAEA_MIN_DENOMINATOR = 1e-12


def normalise_longitude(lon: np.ndarray | float) -> np.ndarray:
    """Wrap longitudes into the -180 to 180 degree range.

    Args:
        lon: One longitude in degrees, or an array of them.

    Returns:
        longitudes: The wrapped longitudes as a float array.
    """
    return (np.asarray(lon, dtype=float) + 180.0) % 360.0 - 180.0


def longitude_stretch(lat: float) -> float:
    """Return how much a degree of longitude shrinks at one latitude.

    Args:
        lat: The latitude in degrees.

    Returns:
        cosine: The cosine of the latitude, never below MIN_COSINE.
    """
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
    if raw > 0.0:
        return raw
    return raw % TURN


def bbox_centre(
    min_lat: float, max_lat: float, west_lon: float, east_lon: float
) -> tuple[float, float]:
    """Return the centre of a feature bounding box.

    Args:
        min_lat: The southernmost latitude in degrees.
        max_lat: The northernmost latitude in degrees.
        west_lon: The westernmost longitude in degrees.
        east_lon: The easternmost longitude in degrees.

    Returns:
        longitude: The centre longitude in -180 to 180 degrees.
        latitude: The centre latitude in degrees.
    """
    centre_lon = west_lon + longitude_span(west_lon, east_lon) / 2.0
    return float(normalise_longitude(centre_lon)), (min_lat + max_lat) / 2.0


def bbox_ring(
    min_lat: float, max_lat: float, west_lon: float, east_lon: float, step: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return a densified closed lon/lat ring tracing a bounding box.

    Args:
        min_lat: The southernmost latitude in degrees.
        max_lat: The northernmost latitude in degrees.
        west_lon: The westernmost longitude in degrees.
        east_lon: The easternmost longitude in degrees.
        step: The longest segment the ring is densified to, in degrees.

    Returns:
        longitudes: The ring longitudes, closed back onto the first point.
        latitudes: The ring latitudes, closed the same way.
    """
    span = longitude_span(west_lon, east_lon)
    east = west_lon + span
    rise = max_lat - min_lat
    along = np.linspace(west_lon, east, max(2, math.ceil(span / step) + 1))
    up = np.linspace(min_lat, max_lat, max(2, math.ceil(rise / step) + 1))
    lons = np.concatenate(
        [along, np.full(up.size, east), along[::-1], np.full(up.size, west_lon)]
    )
    lats = np.concatenate(
        [np.full(along.size, min_lat), up, np.full(along.size, max_lat), up[::-1]]
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


def local_radius_m(lat: np.ndarray | float) -> np.ndarray:
    """Return the radius Mars stands at, at one planetocentric latitude.

    Args:
        lat: The latitude in degrees, or an array of them.

    Returns:
        radius: The radius in metres, falling from the equatorial axis to the polar one.
    """
    phi = np.radians(np.asarray(lat, dtype=float))
    return (EQUATORIAL_RADIUS_M * POLAR_RADIUS_M) / np.hypot(
        POLAR_RADIUS_M * np.cos(phi), EQUATORIAL_RADIUS_M * np.sin(phi)
    )


def haversine_steps(
    lon: np.ndarray, lat: np.ndarray, radius: np.ndarray | float = RADIUS_M
) -> np.ndarray:
    """Return the great-circle distance between each neighbouring pair of points.

    Args:
        lon: The point longitudes in degrees.
        lat: The point latitudes in degrees.
        radius: The sphere each pair is measured on, one radius for all of them or
            one per pair.

    Returns:
        steps: One distance in metres per neighbouring pair, and nothing for fewer than
            two points.
    """
    lam = np.radians(np.asarray(lon, dtype=float))
    phi = np.radians(np.asarray(lat, dtype=float))
    hav = (
        np.sin(np.diff(phi) / 2.0) ** 2
        + np.cos(phi[:-1]) * np.cos(phi[1:]) * np.sin(np.diff(lam) / 2.0) ** 2
    )
    return 2.0 * radius * np.arcsin(np.sqrt(np.clip(hav, 0.0, 1.0)))


def haversine_length(lon: np.ndarray, lat: np.ndarray) -> float:
    """Return the great-circle length along a sequence of lon/lat points.

    Args:
        lon: The point longitudes in degrees.
        lat: The point latitudes in degrees.

    Returns:
        length: The summed length in metres, or 0.0 for fewer than two points.
    """
    return float(haversine_steps(lon, lat).sum())


def northward_m(degrees: float) -> float:
    """Return how far north a span of latitude reaches, in metres.

    Args:
        degrees: The span of latitude in degrees.

    Returns:
        metres: The distance in metres along a meridian.
    """
    return math.radians(degrees) * RADIUS_M


def eastward_m(degrees: float, lat: float) -> float:
    """Return how far east a span of longitude reaches at one latitude, in metres.

    Args:
        degrees: The span of longitude in degrees.
        lat: The latitude it is spanned at, in degrees.

    Returns:
        metres: The distance in metres along that parallel.
    """
    return math.radians(degrees) * RADIUS_M * math.cos(math.radians(lat))


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
    radius: float = RADIUS_M,
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
    radius: float = RADIUS_M,
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
