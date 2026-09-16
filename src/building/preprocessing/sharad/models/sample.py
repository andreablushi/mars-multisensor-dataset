"""One SHARAD track cut to the tile it was kept for."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from building.preprocessing.common.models.sample import Sample


@dataclass(frozen=True, slots=True, kw_only=True)
class SharadSample(Sample):
    """The echoes one track sounded over one tile.

    Attributes:
        power: Delay samples by traces, holding only the traces that are left.
        traces: Which of the original radargram columns these traces are,
            counted from zero.
    """

    power: np.ndarray
    traces: np.ndarray
