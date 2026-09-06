"""One SHARAD track cut to the feature it was kept for."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from building.preprocessing.common.models.sample import Sample


@dataclass(frozen=True, slots=True, kw_only=True)
class SharadSample(Sample):
    """The echoes one track sounded over one feature.

    Attributes:
        power: Delay samples by traces, holding only the traces that are left.
        geometry: One row per kept trace, in the same order, which the altitude
            the delay axis is read through is measured off.
        traces: Which of the original radargram columns these traces are,
            counted from zero.
    """

    power: np.ndarray
    geometry: np.recarray
    traces: np.ndarray
