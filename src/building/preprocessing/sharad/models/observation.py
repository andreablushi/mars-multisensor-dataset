"""One SHARAD radargram as it comes off disk, with its geometry joined onto it."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Which geometry field places a trace.
PLACEMENT = {"latitude": "LATITUDE", "longitude": "LONGITUDE"}

# Which fields the height above ground is read between, in km, for the delay axis.
RADII = {"ground": "MARS RADIUS", "spacecraft": "SPACECRAFT RADIUS"}


@dataclass(frozen=True, slots=True)
class SharadObservation:
    """One track holding only the traces its geometry places.

    Attributes:
        label: What every product it was published as says about it, merged.
        identifier: The observation id.
        power: Delay samples by traces, holding only the placed traces.
        geometry: One row per kept trace, in the same order.
        traces: Which of the original radargram columns these traces are,
            counted from zero.
    """

    identifier: str
    label: dict[str, str]
    power: np.ndarray
    geometry: np.recarray
    traces: np.ndarray

    # A sounder walks a line, so every trace carries its own geometry's pair.
    separable = False

    @property
    def latitude(self) -> np.ndarray:
        """Return the latitude every kept trace was sounded at.

        Returns:
            One per trace, in degrees.
        """
        return self.geometry[PLACEMENT["latitude"]]

    @property
    def longitude(self) -> np.ndarray:
        """Return the longitude every kept trace was sounded at.

        Returns:
            One per trace, in degrees.
        """
        return self.geometry[PLACEMENT["longitude"]]
