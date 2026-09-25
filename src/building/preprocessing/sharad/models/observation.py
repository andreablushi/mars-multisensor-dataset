"""One SHARAD track as it comes off disk, with its geometry joined onto it."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The field the geometry names each radargram column in, counted from one.
COLUMN_FIELD = "RADARGRAM COLUMN"

# Which geometry field places a trace.
LATITUDE_FIELD = "LATITUDE"
LONGITUDE_FIELD = "LONGITUDE"

# Which of them says where the Sun and the spacecraft stood over it.
SOLAR_ZENITH_FIELD = "SZA"
MARS_RADIUS_FIELD = "MARS RADIUS"
SPACECRAFT_RADIUS_FIELD = "SPACECRAFT RADIUS"


@dataclass(frozen=True, slots=True)
class SharadObservation:
    """One track holding only the traces its geometry places.

    Attributes:
        identifier: The observation id.
        label: What every product it was published as says about it, merged.
        power: Delay samples by traces, holding only the placed traces.
        clutter: The simulated clutter power per column, zero without echo.
        geometry: One row per kept trace, in the same order.
        traces: Which original radargram columns these traces are, from zero.
    """

    identifier: str
    label: dict[str, str]
    power: np.ndarray
    clutter: np.ndarray
    geometry: np.recarray
    traces: np.ndarray
