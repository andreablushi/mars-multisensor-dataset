"""How far round its own year Mars had turned, which is what a window is held to."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime

# The Julian date the Unix epoch falls on, which every timestamp is counted from
_JD_UNIX_EPOCH = 2440587.5

# Seconds in a day, which the Julian date below is counted in
_DAY_SECONDS = 86400.0

# The Julian date of the J2000 epoch the series below is written against
_J2000 = 2451545.0


def arcs(when: Sequence[datetime]) -> list[float]:
    """Say how far round its orbit Mars had come as each observation was taken.

    Args:
        when: The moments to place, each read as UTC.

    Returns:
        arcs: The angle Mars had swept by each, in degrees, one to each moment given.
    """
    swept: list[float] = []
    for moment in when:
        offset = moment.timestamp() / _DAY_SECONDS + _JD_UNIX_EPOCH - _J2000
        # Allison and McEwen (1997), less its perturbers and terrestrial time
        anomaly = math.radians(19.3871 + 0.52402073 * offset)
        # Where the true sun stands against the mean one, on Mars' eccentric orbit
        centre = (
            10.691 * math.sin(anomaly)
            + 0.623 * math.sin(2.0 * anomaly)
            + 0.050 * math.sin(3.0 * anomaly)
            + 0.005 * math.sin(4.0 * anomaly)
        )
        swept.append(270.3871 + 0.524038496 * offset + centre)
    return swept
