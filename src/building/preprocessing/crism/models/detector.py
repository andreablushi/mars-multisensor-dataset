"""One detector's half of a CRISM observation, as the correction chain holds it."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from building.preprocessing.crism.models.mask import Mask


@dataclass(frozen=True, slots=True)
class Detector:
    """One detector's half of an observation, and what its cube holds.

    Attributes:
        name: Which detector, `l` for infrared or `s` for visible.
        cube: The values as lines by samples by bands, its bands ascending in
            wavelength. I/F with its uncalibrated columns and bands NaN as
            `preprocess.read_detectors` returns it, and a ratio against each
            column's own median, those cells zero, once the chain has been
            through it.
        wavelengths: The centre wavelength in nm of every column and band, as
            columns by bands, in the same order as the cube. Columns and bands
            the detector was never calibrated for hold NaN. One centre per band
            is `bands_calibration.centres` of this, not stored beside it.
        mask: Where the cube was filled rather than measured, once it has been
            cleaned, and None while it is still as it was read.
    """

    name: str
    cube: np.ndarray
    wavelengths: np.ndarray
    mask: Mask | None = None
