"""One SHARAD track cut to the tile it was kept for."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mars_multisensor_dataset.building.preprocessing.common.models.sample import Sample


@dataclass(frozen=True, slots=True, kw_only=True)
class SharadSample(Sample):
    """The echoes one track sounded over one tile.

    Attributes:
        power: dB averaged along each delay row and soft gated at its noise
            threshold, delay samples by traces, only the traces left.
        clutter: The simulated clutter in dB on the same grid, floored without echo
            and aligned on its own average surface over the traces this crop
            measured.
        traces: Which original radargram columns these traces are, from zero.
        noise_threshold_db: The threshold its noise was gated at, in dB.
        noise_spread_db: The standard deviation of that noise in dB.
        surface_offset_db: The average surface peak of the traces it measured, in
            dB, which aligns the radargram on its surface once subtracted.
    """

    power: np.ndarray
    clutter: np.ndarray
    traces: np.ndarray
    noise_threshold_db: float
    noise_spread_db: float
    surface_offset_db: float
