"""One CTX scan calibrated by ISIS, with where its sampled lines fall."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class CtxObservation:
    """One calibrated scan, still in camera geometry.

    Attributes:
        identifier: The observation id.
        label: What ODE says about it, the geometry it was taken at.
        cube: The calibrated ISIS cube, projected one tile at a time.
        lines: How many lines the scan holds.
        line: The line of every sampled point, counted from one.
        latitude: The planetocentric latitude of every sampled point.
        longitude: The positive east longitude of every sampled point, 0 to 360.
    """

    identifier: str
    label: dict[str, str]
    cube: Path
    lines: int
    line: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
