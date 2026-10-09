"""One CRISM observation as it comes off disk, and the detector halves it joins."""

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
class DetectorCube:
    """One detector's half of an observation, cleaned, on the bands it keeps.

    Attributes:
        cube: The values as lines by samples by bands, bands ascending.
        table: The centre wavelength in nm per column and band.
        columns: One flag per sample, True where the column was never calibrated.
        pixels: Lines by samples, True where the pixel has no usable spectrum.
    """

    cube: np.ndarray
    table: np.ndarray
    columns: np.ndarray
    pixels: np.ndarray


@dataclass(frozen=True, slots=True)
class CrismObservation:
    """One observation with its two detectors joined.

    Attributes:
        label: What every product it was published as says about it, merged.
        cube: Lines by columns by the survey's band grid, NaN for unmeasured bands.
        geometry: The backplanes on the same grid, as lines by columns by 14.
        valid: Lines by columns, True where the pixel is a measurement.
        measured_bands: One flag per band of that grid this observation measured.
    """

    label: dict[str, str]
    cube: np.ndarray
    geometry: np.ndarray
    valid: np.ndarray
    measured_bands: np.ndarray
