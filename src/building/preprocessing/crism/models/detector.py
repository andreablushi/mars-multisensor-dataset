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
            wavelength and its uncalibrated columns and bands NaN.
        wavelengths: The centre wavelength in nm of every column and band, as
            columns by bands in the cube's own order, NaN where uncalibrated.
        mask: Where the cube was filled rather than measured, once it has been
            cleaned, and None while it is still as it was read.
    """

    name: str
    cube: np.ndarray
    wavelengths: np.ndarray
    mask: Mask | None = None
