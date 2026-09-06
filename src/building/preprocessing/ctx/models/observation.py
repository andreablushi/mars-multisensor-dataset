"""One CTX scan as it comes off disk, placed on its own grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class CtxObservation:
    """One scan on the grid its label projects it onto.

    Attributes:
        label: What every product it was published as says about it, merged.
        identifier: The observation id.
        image: The brightness as lines by samples.
        latitude: The centre latitude in degrees of every line.
        longitude: The centre longitude in degrees of every sample.
    """

    identifier: str
    label: dict[str, str]
    image: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray

    # An RDR is projected onto a regular grid, so one axis places each side.
    separable = True
