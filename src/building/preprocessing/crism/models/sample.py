"""One CRISM observation cut to the feature it was kept for."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from building.preprocessing.common.models.sample import Sample


@dataclass(frozen=True, slots=True, kw_only=True)
class CrismSample(Sample):
    """The spectra one observation measured over one feature.

    Attributes:
        cube: Lines by columns by bands, bands ascending in wavelength.
        wavelengths: The centre wavelength of every column and band, in that
            same order.
        columns: Which of the original 64 samples these columns are.
    """

    cube: np.ndarray
    wavelengths: np.ndarray
    columns: np.ndarray
