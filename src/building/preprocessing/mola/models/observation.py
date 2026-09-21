"""The sheets one tile stands on, laid onto the single equatorial grid they share."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class MolaObservation:
    """The height over one box, on the sheets' own grid of latitude and longitude.

    Attributes:
        label: What the products it was read from say about it, merged.
        identifier: The grid it was read from, which its crops are stored under.
        topography: The height above the areoid in metres, lines by samples.
        down: The latitude of every line in degrees, falling southward.
        across: The longitude of every sample, rising eastward past a turn.
    """

    identifier: str
    label: dict[str, str]
    topography: np.ndarray
    down: np.ndarray
    across: np.ndarray

    # Either projection is regular on both axes, so one axis places each side.
    separable = True
