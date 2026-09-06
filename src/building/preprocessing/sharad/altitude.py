"""How high the spacecraft flew over one track, which its delay axis is read through."""

from __future__ import annotations

from building.preprocessing.sharad.models.observation import RADII
from building.preprocessing.sharad.models.sample import SharadSample

# The archive writes both radii in kilometres.
KM = 1000.0


def altitude_m(sample: SharadSample) -> tuple[float, float]:
    """Return how low and how high the spacecraft was above the ground.

    The delay axis is not turned into a depth here, since that needs a
    dielectric constant the subsurface is assumed to have, which is a choice
    about the ground rather than about where the track ran. This keeps what
    that conversion needs beside the track that would be converted.

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
