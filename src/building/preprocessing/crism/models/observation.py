"""One CRISM observation as it comes off disk, both detectors as a single cube."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Which DDR backplane places a pixel; only the slope and the height say what MOLA does.
LATITUDE_PLANE = 3
LONGITUDE_PLANE = 4

ACQUISITION_PLANES = {
    "incidence_deg": 0,
    "emission_deg": 1,
    "phase_deg": 2,
    "local_solar_time_h": 12,
}


@dataclass(frozen=True, slots=True)
class CrismObservation:
    """One observation with its two detectors joined.

    Attributes:
        identifier: The observation id.
        label: What every product it was published as says about it, merged.
        cube: Lines by columns by the survey's band grid, NaN for unmeasured bands.
        geometry: The backplanes on the same grid, as lines by columns by 14.
        valid: Lines by columns, True where the pixel is a measurement.
        measured_bands: One flag per band of that grid this observation measured.
    """

    identifier: str
    label: dict[str, str]
    cube: np.ndarray
    geometry: np.ndarray
    valid: np.ndarray
    measured_bands: np.ndarray

    @property
    def latitude(self) -> np.ndarray:
        """Return the latitude of every pixel, lines by columns in degrees."""
        return self.geometry[:, :, LATITUDE_PLANE]

    @property
    def longitude(self) -> np.ndarray:
        """Return the longitude of every pixel, lines by columns in degrees."""
        return self.geometry[:, :, LONGITUDE_PLANE]
