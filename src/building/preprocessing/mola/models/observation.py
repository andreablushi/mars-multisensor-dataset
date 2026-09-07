"""The tiles one feature stands on, laid onto the single cylindrical grid they share."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class MolaObservation:
    """The height over one box, on the tiles' own grid of latitude and longitude.

    Attributes:
        label: What the products it was read from say about it, merged.
        identifier: The grid it was read from, which every crop of it is
            stored under.
        topography: The height of the ground above the areoid in metres, as
            lines by samples, interpolated where no shot fell in the bin.
        down: The latitude of every line in degrees, falling southward.
        across: The longitude of every sample, rising eastward and running past
            a whole turn where the box crosses the meridian.
    """

    identifier: str
    label: dict[str, str]
    topography: np.ndarray
    down: np.ndarray
    across: np.ndarray

    # Either projection is regular on both axes, so one axis places each side.
    separable = True
