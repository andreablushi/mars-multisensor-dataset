"""What one detector's cube holds that is not a measurement."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Mask:
    """Where one cube was filled rather than measured, and why.

    Attributes:
        columns: One flag per sample, True where the column was never calibrated.
        bands: One flag per band, True where the band is not kept.
        pixels: Lines by samples, True where the pixel has no usable spectrum.
        fill: The value every flagged cell was replaced with.
    """

    columns: np.ndarray
    bands: np.ndarray
    pixels: np.ndarray
    fill: float
