"""Where every delay sample of one track stands, above the areoid it is posted from.

The US radargram is posted so that its centre range cell carries the free-space
round-trip delay of the MOLA defined areoid, and its samples are 0.0375
microseconds apart, so one sample is a fixed step of free-space range and the
axis this reads is the same for every trace of every track. That makes it the
datum MOLA already publishes its topography against, and it costs no assumption
about the ground: an echo from below the surface stands at the elevation it
would have in vacuum, which is deeper than the reflector by the square root of
the dielectric constant the ground is later taken to have.
"""

from __future__ import annotations

import numpy as np

SAMPLE_INTERVAL_S = 0.0375e-6
LIGHT_SPEED_M_S = 299792458.0
SAMPLE_RANGE_M = LIGHT_SPEED_M_S * SAMPLE_INTERVAL_S / 2.0


def elevation_m(samples: int) -> np.ndarray:
    """Return how high above the areoid every delay sample of a track stands.

    Args:
        samples: How many delay samples one trace holds, whose centre cell is
            the areoid the whole window is posted from.

    Returns:
        One height in metres per delay sample, falling as the delay grows.
    """
    return -(np.arange(samples) - (samples // 2 - 1)) * SAMPLE_RANGE_M
