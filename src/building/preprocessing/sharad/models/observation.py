"""One SHARAD radargram as it comes off disk, with its geometry joined onto it."""

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
    """One track holding only the traces its geometry places.

    Attributes:
        label: What every product it was published as says about it, merged.
        identifier: The observation id.
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

    @property
    def solar_zenith_deg(self) -> np.ndarray:
        """Return how far off the vertical the Sun stood over every kept trace.

        Returns:
            zenith: One per trace, in degrees, the incidence a camera would see.
        """
        return self.geometry[SOLAR_ZENITH_FIELD]

    @property
    def spacecraft_altitude_km(self) -> np.ndarray:
        """Return how far above the ground the spacecraft flew over every trace.

        Returns:
            altitude: One per trace, in km, the difference of the two radii.
        """
        return self.geometry[SPACECRAFT_RADIUS_FIELD] - self.geometry[MARS_RADIUS_FIELD]
