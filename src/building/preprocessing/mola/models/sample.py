"""One MOLA tile cut to the feature it was kept for."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from building.preprocessing.common.models.sample import Sample


@dataclass(frozen=True, slots=True, kw_only=True)
class MolaSample(Sample):
    """The height one tile holds over one feature.

    Attributes:
        topography: The height of the ground above the areoid in metres, as
            lines by samples.
    """

    topography: np.ndarray
