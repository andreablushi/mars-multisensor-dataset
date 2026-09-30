"""One CTX scan as it comes off disk, calibrated by ISIS, with where its lines fall."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class ScanStatistics:
    """What every crop of one scan is clipped and standardised by.

    Attributes:
        low: The count the scan's darkest pixels are clipped up to.
        high: The count its brightest pixels are clipped down to.
        mean: The mean of the clipped counts.
        std: Their standard deviation.
    """

    low: float
    high: float
    mean: float
    std: float


@dataclass(frozen=True, slots=True)
class CtxObservation:
    """One calibrated scan, still in camera geometry.

    Attributes:
        label: What ODE says about it, the geometry it was taken at.
        cube: The calibrated ISIS cube, projected one tile at a time.
        lines: How many lines the scan holds.
        line: The line of every sampled point, counted from one.
        latitude: The planetocentric latitude of every sampled point.
        longitude: The positive east longitude of every sampled point, 0 to 360.
        statistics: What its whole scan is standardised by.
    """

    label: dict[str, str]
    cube: Path
    lines: int
    line: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    statistics: ScanStatistics
