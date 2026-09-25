"""One detector's half of a CRISM observation, as the correction chain holds it."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from building.configs.crism import Detector
from building.preprocessing.crism.models.mask import Mask


@dataclass(frozen=True, slots=True)
class DetectorCube:
    """One detector's half of an observation, and what its cube holds.

    Attributes:
        name: Which detector, `l` for infrared or `s` for visible.
        cube: The values as lines by samples by bands, bands ascending.
        wavelengths: The centre wavelength in nm per column and band.
        mask: Where the cube was filled once cleaned, or None before.
    """

    name: Detector
    cube: np.ndarray
    wavelengths: np.ndarray
    mask: Mask | None = None
