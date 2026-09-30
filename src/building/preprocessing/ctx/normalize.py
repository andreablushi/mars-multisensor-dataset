"""Standardising a CTX scan by the counts its whole calibrated cube holds."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import tifffile

from building.preprocessing.ctx.isis import export_image
from building.preprocessing.ctx.models.observation import ScanStatistics
from building.preprocessing.ctx.models.sample import BLANK


def scan_statistics(cube: Path, clip: Sequence[float], stride: int) -> ScanStatistics:
    """Return the clip bounds and clipped mean and std of one whole scan.

    Args:
        cube: The calibrated scan, in camera geometry.
        clip: The low and high percentiles its counts are clipped at.
        stride: How many lines and samples apart the counts it is measured on sit.

    Returns:
        statistics: What every crop of the scan is standardised by.

    Raises:
        RuntimeError: When isis2std fails.
    """
    image = cube.with_suffix(".tif")
    try:
        export_image(cube, image)
        every = slice(None, None, stride)
        sampled = tifffile.imread(image, selection=(every, every))
    finally:
        image.unlink(missing_ok=True)
    held = sampled[sampled != BLANK].astype(np.float64)
    low, high = np.percentile(held, clip)
    clipped = np.clip(held, low, high)
    return ScanStatistics(
        float(low), float(high), float(clipped.mean()), float(clipped.std())
    )


def normalized_pixels(
    pixels: np.ndarray, measured: np.ndarray, statistics: ScanStatistics
) -> np.ndarray:
    """Return one crop's counts clipped and standardised by its whole scan.

    Args:
        pixels: The crop's counts, as lines by samples.
        measured: Where the crop holds a measurement, on the same grid.
        statistics: What the whole scan is standardised by.

    Returns:
        normalized: The standardised counts, zero where nothing was measured.
    """
    clipped = np.clip(pixels, statistics.low, statistics.high).astype(np.float32)
    return np.where(measured, (clipped - statistics.mean) / statistics.std, 0.0)
