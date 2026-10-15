"""One SHARAD track cut to the tile it was kept for."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from building.preprocessing.common.models.sample import Sample


@dataclass(frozen=True, slots=True, kw_only=True)
class SharadSample(Sample):
    """The echoes one track sounded over one tile.

    Attributes:
        power: dB aligned on the average surface, delay samples by traces, only
            the traces left.
        clutter: The simulated clutter in dB on the same grid, floored without echo
            and aligned on its own average surface.
        traces: Which original radargram columns these traces are, from zero.
    """

    power: np.ndarray
    clutter: np.ndarray
    traces: np.ndarray
