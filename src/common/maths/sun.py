"""Where the Sun stands over Mars at a moment, by Mars24 (Allison and McEwen 2000)."""

from __future__ import annotations

import math
from datetime import datetime

from common.maths.geodesy import TURN

JD_UNIX_EPOCH = 2440587.5

J2000 = 2451545.0

SECONDS_PER_DAY = 86400.0

TT_MINUS_UTC_S = 69.184


def julian_date_tt(utc: datetime) -> float:
    """Return the terrestrial time Julian date of one UTC moment."""
    return JD_UNIX_EPOCH + (utc.timestamp() + TT_MINUS_UTC_S) / SECONDS_PER_DAY


def solar_longitude_and_centre(jd_tt: float) -> tuple[float, float]:
    """Return the areocentric longitude of the Sun and the equation of centre.

    Args:
        jd_tt: The moment, as a terrestrial time Julian date.

    Returns:
        ls: The areocentric longitude of the Sun, in degrees, 0 to 360.
        centre: How far the true Sun stands from the mean one, in degrees.
    """
    days = jd_tt - J2000
    mean_anomaly = math.radians(19.3871 + 0.52402073 * days)
    fictitious_sun = 270.3871 + 0.524038496 * days
    centre = (
        (10.691 + 3.0e-7 * days) * math.sin(mean_anomaly)
        + 0.623 * math.sin(2 * mean_anomaly)
        + 0.050 * math.sin(3 * mean_anomaly)
        + 0.005 * math.sin(4 * mean_anomaly)
        + 0.0005 * math.sin(5 * mean_anomaly)
    )
    return (fictitious_sun + centre) % TURN, centre


def solar_longitude_deg(utc: datetime) -> float:
    """Return the areocentric longitude of the Sun at one UTC moment, 0 to 360."""
    return solar_longitude_and_centre(julian_date_tt(utc))[0]


def subsolar_point_deg(utc: datetime) -> tuple[float, float]:
    """Return where the Sun stands at the zenith at one UTC moment.

    Args:
        utc: The moment.

    Returns:
        longitude: The subsolar east longitude in degrees, 0 to 360.
        latitude: The subsolar planetographic latitude in degrees.
    """
    jd_tt = julian_date_tt(utc)
    ls, centre = solar_longitude_and_centre(jd_tt)
    ls_rad = math.radians(ls)
    sine = math.sin(ls_rad)
    equation_of_time = (
        2.861 * math.sin(2 * ls_rad)
        - 0.071 * math.sin(4 * ls_rad)
        + 0.002 * math.sin(6 * ls_rad)
        - centre
    )
    mean_solar_time_h = (
        24 * ((jd_tt - 2451549.5) / 1.0274912517 + 44796.0 - 0.0009626)
    ) % 24
    west = mean_solar_time_h * TURN / 24 + equation_of_time + TURN / 2
    latitude = math.degrees(math.asin(0.42565 * sine)) + 0.25 * sine
    return -west % TURN, latitude


def solar_azimuth_deg(utc: datetime, lat: float, lon: float) -> float:
    """Return where the Sun stands seen from one place, clockwise from north.

    Args:
        utc: The moment.
        lat: The place's latitude in degrees.
        lon: The place's east longitude in degrees.

    Returns:
        azimuth: The bearing from the place to the subsolar point, 0 to 360.
    """
    sun_lon, sun_lat = subsolar_point_deg(utc)
    phi, sun_phi = math.radians(lat), math.radians(sun_lat)
    apart = math.radians(sun_lon - lon)
    bearing = math.atan2(
        math.sin(apart) * math.cos(sun_phi),
        math.cos(phi) * math.sin(sun_phi)
        - math.sin(phi) * math.cos(sun_phi) * math.cos(apart),
    )
    return math.degrees(bearing) % TURN
