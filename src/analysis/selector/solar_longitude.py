"""The solar longitude Mars had reached at a moment, which a window is held to."""

from __future__ import annotations

import math

# The Julian date the Unix epoch falls on, which every timestamp is counted from
_JD_UNIX_EPOCH = 2440587.5

# The Julian date of the J2000 epoch the series below is written against
_J2000 = 2451545.0


def solar_longitude(day: float) -> float:
    """Return how far round its orbit Mars had come at one moment.

    Args:
        day: The moment, in UTC days since the Unix epoch.

    Returns:
        ls: The angle Mars had swept by then, in degrees.
    """
    offset = day + _JD_UNIX_EPOCH - _J2000
    # Allison and McEwen (1997), less its perturbers and terrestrial time
    anomaly = math.radians(19.3871 + 0.52402073 * offset)
    # Where the true sun stands against the mean one, on Mars' eccentric orbit
    centre = (
        10.691 * math.sin(anomaly)
        + 0.623 * math.sin(2.0 * anomaly)
        + 0.050 * math.sin(3.0 * anomaly)
        + 0.005 * math.sin(4.0 * anomaly)
    )
    return 270.3871 + 0.524038496 * offset + centre
