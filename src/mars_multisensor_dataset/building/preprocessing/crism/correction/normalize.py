"""Dividing each spectrum by its L2 norm, keeping its shape but not its brightness."""

from __future__ import annotations

import numpy as np


def normalized_spectra(cube: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Return every measured spectrum divided by its own L2 norm, as crism_ml's normr.

    Args:
        cube: The values as lines by samples by bands, NaN for unmeasured bands.
        valid: Lines by samples, True where the pixel is a measurement.

    Returns:
        normalized: The same cube, every measured spectrum of unit length.
    """
    norm = np.linalg.norm(np.nan_to_num(cube), axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(valid[..., None], cube / norm, cube).astype(cube.dtype)
