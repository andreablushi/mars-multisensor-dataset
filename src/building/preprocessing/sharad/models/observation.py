"""One SHARAD track as it comes off disk, with its geometry joined onto it."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Which geometry field places a trace.
LATITUDE_FIELD = "LATITUDE"
LONGITUDE_FIELD = "LONGITUDE"

# Which of them says where the Sun and the spacecraft stood over it.
SOLAR_ZENITH_FIELD = "SZA"
MARS_RADIUS_FIELD = "MARS RADIUS"
SPACECRAFT_RADIUS_FIELD = "SPACECRAFT RADIUS"


@dataclass(frozen=True, slots=True)
class SharadObservation:
    """One track with its geometry joined onto it, a row per trace.

    Attributes:
        label: What every product it was published as says about it, merged.
        power: Normalized dB, delay samples by traces.
        clutter: The simulated clutter power per column, zero without echo.
        geometry: One row per trace, in the same order.
    """

    label: dict[str, str]
    power: np.ndarray
    clutter: np.ndarray
    geometry: np.recarray
