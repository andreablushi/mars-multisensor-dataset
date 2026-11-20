"""Levelling the columns of a strip, across it and along it, and removing spikes."""

from __future__ import annotations

import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from mars_multisensor_dataset.building.configs.crism import FILL
from mars_multisensor_dataset.building.models.settings import SpikeSettings


def flat_field(cube: np.ndarray, refused: np.ndarray) -> None:
    """Scale each column by the strip's median over the column's, band by band.

    Args:
        cube: The values as lines by samples by bands, levelled in place.
        refused: Lines by samples, True where the pixel is not a measurement.
    """
    strip = np.median(cube[~refused], axis=0)
    for at in range(cube.shape[1]):
        live = ~refused[:, at]
        # A column with no measurement has nothing to level, and is refused anyway.
        if live.any():
            column = cube[:, at, :]
            held = column[live]
            column[live] = held * (strip / np.median(held, axis=0))
    cube[refused] = FILL


def remove_spikes(cube: np.ndarray, refused: np.ndarray, spikes: SpikeSettings) -> None:
    """Remove spikes in repeated passes over a moving median, as crism_ml does.

    Args:
        cube: The flat-fielded values as lines by samples by bands, changed in place.
        refused: Lines by samples, True where the pixel is not a measurement.
        spikes: The passes, the median's width in bands, truncated at the ends, and
            how many spreads from the median a value may sit.
    """
    # A refused pixel is flat, so leaving it in would pull the threshold down.
    live = ~refused
    # The median, the distance from it and what that catches, refilled each pass.
    median = np.empty_like(cube)
    apart = np.empty_like(cube)
    caught = np.empty(cube.shape, dtype=bool)
    left, right = spikes.window // 2, spikes.window - spikes.window // 2
    for _ in range(spikes.passes):
        for at in range(cube.shape[-1]):
            near = cube[..., max(at - left, 0) : at + right]
            held = near.shape[-1]
            # Partitioned, not sorted, which stops at the middle and writes in place.
            part = np.partition(near, held // 2, axis=-1)
            if held % 2:
                median[..., at] = part[..., held // 2]
            else:
                median[..., at] = 0.5 * (
                    part[..., held // 2 - 1] + part[..., held // 2]
                )
        np.subtract(median, cube, out=apart)
        np.abs(apart, out=apart)
        # crism_ml judges every sample against the measured cube's own spread.
        limit = np.mean(apart.mean(axis=-1), where=live) + spikes.sigma * np.mean(
            apart.std(ddof=1, axis=-1), where=live
        )
        np.greater(apart, limit, out=caught)
        np.copyto(cube, median, where=caught)


def level_columns(cube: np.ndarray, valid: np.ndarray, lines: int) -> None:
    """Scale each column by the strip's median over its running median, band by band.

    Args:
        cube: The values as lines by samples by bands, levelled in place.
        valid: Lines by samples, True where the pixel is a measurement.
        lines: How many lines the running median spans along the track, odd.
    """
    held = np.where(valid[..., None], cube, np.nan).astype("f4")
    half = lines // 2
    padded = np.pad(held, ((half, half), (0, 0), (0, 0)), constant_values=np.nan)
    with warnings.catch_warnings(action="ignore"):
        strip = np.nanmedian(held[valid], axis=0)
        for at in np.flatnonzero(np.isfinite(strip)):
            local = np.nanmedian(
                sliding_window_view(padded[..., at], lines, axis=0), -1
            )
            band = cube[..., at]
            band[valid] *= strip[at] / local[valid]
