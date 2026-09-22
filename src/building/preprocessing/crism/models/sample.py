"""One CRISM observation cut to the tile it was kept for."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from building.preprocessing.common.models.sample import Sample


@dataclass(frozen=True, slots=True, kw_only=True)
class CrismSample(Sample):
    """The spectra one observation measured over one tile.

    Attributes:
        cube: Lines by columns by the survey's band grid, NaN for unmeasured bands.
    """

    cube: np.ndarray
