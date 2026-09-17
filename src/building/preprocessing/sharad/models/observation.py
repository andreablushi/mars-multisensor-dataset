"""One SHARAD radargram as it comes off disk, with its geometry joined onto it."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Which geometry field places a trace.
LATITUDE_FIELD = "LATITUDE"
LONGITUDE_FIELD = "LONGITUDE"


@dataclass(frozen=True, slots=True)
class SharadObservation:
    """One track holding only the traces its geometry places.

    Attributes:
        label: What every product it was published as says about it, merged.
        identifier: The observation id.
        power: Delay samples by traces, holding only the placed traces.
        clutter: The simulated surface clutter power on the same grid, zero where
            no surface echo is predicted.
        geometry: One row per kept trace, in the same order.
        traces: Which of the original radargram columns these traces are,
            counted from zero.
    """

    identifier: str
    label: dict[str, str]
    power: np.ndarray
    clutter: np.ndarray
    geometry: np.recarray
    traces: np.ndarray

    # A sounder walks a line, so every trace carries its own geometry's pair.
    separable = False

    @property
    def latitude(self) -> np.ndarray:
        """Return the latitude every kept trace was sounded at.

        Returns:
            latitude: One per trace, in degrees.
        """
        return self.geometry[LATITUDE_FIELD]

    @property
    def longitude(self) -> np.ndarray:
        """Return the longitude every kept trace was sounded at.

        Returns:
            longitude: One per trace, in degrees.
        """
        return self.geometry[LONGITUDE_FIELD]
