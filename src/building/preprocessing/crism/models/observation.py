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
class Mask:
    """Where one cube was filled rather than measured, and why.

    Attributes:
        columns: One flag per sample, True where the column was never calibrated.
        bands: One flag per band, True where the band is not kept.
        pixels: Lines by samples, True where the pixel has no usable spectrum.
        fill: The value every flagged cell was replaced with.
    """

    columns: np.ndarray
    bands: np.ndarray
    pixels: np.ndarray
    fill: float


@dataclass(frozen=True, slots=True)
class DetectorCube:
    """One detector's half of an observation, and what its cube holds.

    Attributes:
        cube: The values as lines by samples by bands, bands ascending.
        table: The centre wavelength in nm per column and band.
        mask: Where the cleaning filled the cube rather than kept a measurement.
    """

    cube: np.ndarray
    table: np.ndarray
    mask: Mask


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
