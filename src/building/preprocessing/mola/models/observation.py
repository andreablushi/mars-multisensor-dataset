"""One MOLA tile on the grid its label projects it onto."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class MolaObservation:
    """One tile, its height on the grid its own label places it on.

    Attributes:
        label: What the product it was published as says about it.
        identifier: The tile id.
        topography: The height of the ground above the areoid in metres, as
            lines by samples, interpolated where no shot fell in the bin.
        latitude: The centre latitude in degrees of every line.
        longitude: The centre longitude in degrees of every sample.
    """

    identifier: str
    label: dict[str, str]
    topography: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray

    # A gridded tile is simple cylindrical, so one axis places each side.
    separable = True
