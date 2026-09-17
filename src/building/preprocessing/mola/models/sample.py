"""One MOLA sheet cut to the tile it was kept for."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from building.preprocessing.common.models.sample import Sample


@dataclass(frozen=True, slots=True, kw_only=True)
class MolaSample(Sample):
    """The height one sheet holds over one tile.

    Attributes:
        delay: The radargram row the ground's nadir echo lands on, as lines by
            samples.
    """

    delay: np.ndarray
