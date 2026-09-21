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
        edges: One flag per band, True where it fell outside the window.
        scattered: How many kept values fell outside a brightness's range.
        pixels: Lines by samples, True where the pixel has no usable spectrum.
        fill: The value every flagged cell was replaced with.
        atmospheric: One flag per band dropped for absorption, or None before.
        stripes: Samples by bands, True where a cell was levelled, or None before.
    """

    columns: np.ndarray
    bands: np.ndarray
    edges: np.ndarray
    scattered: int
    pixels: np.ndarray
    fill: float
    atmospheric: np.ndarray | None = None
    stripes: np.ndarray | None = None
