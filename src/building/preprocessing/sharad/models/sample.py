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
        clutter: The simulated clutter power on the same grid, zero without echo.
        traces: Which original radargram columns these traces are, from zero.
    """

    power: np.ndarray
    clutter: np.ndarray
    traces: np.ndarray
