"""Where a SHARAD crop holds only noise, and what that noise measures."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Noise:
    """The noise of one crop, measured on the crop averaged along each delay row.

    Attributes:
        averaged: The crop averaged along each delay row, in dB, delay samples by
            the traces it measured.
        deep: The first row of the noise below its last echo.
        mask: Where it holds only noise, on the same grid.
        mean: The noise's mean in dB.
        spread: The noise's standard deviation in dB.
        threshold: The dB value its pure noise exceeds at the gate's false-alarm
            rate.
    """

    averaged: np.ndarray
    deep: int
    mask: np.ndarray
    mean: float
    spread: float
    threshold: float
