"""One MOLA sheet cut to the tile it was kept for."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from building.preprocessing.common.models.sample import Sample


@dataclass(frozen=True, slots=True, kw_only=True)
class MolaSample(Sample):
    """The height one sheet holds over one tile.

    Attributes:
        elevation: The height of the ground above the areoid in metres, as lines
            by samples, in the whole metres the record is gridded in.
        delay: The radargram row that height's nadir echo lands on, held to the
            window where it lands off it.
        delay_inside: Which of those rows the window holds, false where the row
            was held to an edge it stands past.
    """

    elevation: np.ndarray
    delay: np.ndarray
    delay_inside: np.ndarray
