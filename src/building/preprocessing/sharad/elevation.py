"""How high a sounding stands: above the areoid, and above the ground below it."""

from __future__ import annotations

import numpy as np

from building.preprocessing.sharad.models.observation import RADII
from building.preprocessing.sharad.models.sample import SharadSample

# The archive writes both radii in kilometres.
KM = 1000.0

# How far apart the archive posts two delay samples of one echo record.
SAMPLE_INTERVAL_S = 0.0375e-6
LIGHT_SPEED_M_S = 299792458.0

# What one sample is worth of free-space range, the centre cell being the areoid.
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


def altitude_m(sample: SharadSample) -> tuple[float, float]:
    """Return how low and how high the spacecraft was above the ground.

    Args:
        sample: The track cut to the feature it was kept for.

    Returns:
        The lowest and the highest height above the ground in metres, over the
        traces the track keeps.
    """
    above = (
        sample.geometry[RADII["spacecraft"]] - sample.geometry[RADII["ground"]]
    ) * KM
    return float(above.min()), float(above.max())
