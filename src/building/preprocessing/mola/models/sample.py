"""One MOLA sheet cut to the tile it was kept for."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from building.preprocessing.common.models.sample import Sample


@dataclass(frozen=True, slots=True, kw_only=True)
class MolaSample(Sample):
    """The height one sheet holds over one tile.

    Attributes:
        elevation: The height above the areoid in whole metres, lines by samples.
        delay: The radargram row the nadir echo lands on, held to the window.
        delay_inside: Which of those rows the window holds.
    """

    elevation: np.ndarray
    delay: np.ndarray
    delay_inside: np.ndarray
