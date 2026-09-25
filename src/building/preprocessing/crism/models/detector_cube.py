"""One detector's half of a CRISM observation, as the correction chain holds it."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from building.preprocessing.crism.models.mask import Mask


@dataclass(frozen=True, slots=True)
class DetectorCube:
    """One detector's half of an observation, and what its cube holds.

    Attributes:
        cube: The values as lines by samples by bands, bands ascending.
        wavelengths: The centre wavelength in nm per column and band.
        mask: Where the cleaning filled the cube rather than kept a measurement.
    """

    cube: np.ndarray
    wavelengths: np.ndarray
    mask: Mask
