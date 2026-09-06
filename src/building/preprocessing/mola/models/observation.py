"""Both planes of one MOLA tile on a single grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class MolaObservation:
    """One tile with its two planes joined onto one grid.

    Attributes:
        label: What every product it was published as says about it, merged.
        identifier: The tile id.
        topography: The height of the ground above the areoid in metres, as
            lines by samples.
        counts: How many shots each bin was measured with, on the same grid,
            zero where the height was interpolated rather than observed.
        latitude: The centre latitude in degrees of every line.
        longitude: The centre longitude in degrees of every sample.
    """

    identifier: str
    label: dict[str, str]
    topography: np.ndarray
    counts: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray

    # A gridded tile is simple cylindrical, so one axis places each side.
    separable = True
