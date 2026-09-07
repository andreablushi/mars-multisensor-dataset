"""What one detector's cube holds that is not a measurement."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Mask:
    """Where one cube was filled rather than measured, and why.

    Attributes:
        columns: One flag per sample, True where the detector was never
            calibrated at that column, so the whole column is unusable.
        bands: One flag per band, True where the band is not kept, whether never
            calibrated or outside the window the detector is trusted over.
        edges: One flag per band, True only where the band was dropped for falling
            outside the window.
        scattered: How many values inside the kept columns and bands were
            outside the range a brightness can take.
        pixels: Lines by samples, True where the pixel carries no usable
            spectrum, being in a dead column or holding a scattered value.
        fill: The value every flagged cell was replaced with.
        atmospheric: One flag per band, True where the band was dropped for falling
            in a window the atmosphere absorbs, and None while that is not done.
        stripes: Samples by bands, True where one detector cell was levelled onto
            its neighbours, and None while the cube has not been destriped.
    """

    columns: np.ndarray
    bands: np.ndarray
    edges: np.ndarray
    scattered: int
    pixels: np.ndarray
    fill: float
    atmospheric: np.ndarray | None = None
    stripes: np.ndarray | None = None
