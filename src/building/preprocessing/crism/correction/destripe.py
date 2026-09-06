"""crism_ml's per-column spike removal, with thresholds set for survey bands."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from building.preprocessing.crism import configs
from building.preprocessing.crism.correction import bands_calibration
from building.preprocessing.crism.models.mask import Mask


def remove_spike_columns(
    cube: np.ndarray, mask: Mask, table: np.ndarray, detector: str
) -> Mask:
    """Replace every band of a column that spikes away from its neighbours.

    Args:
        cube: The values as lines by samples by bands, already masked, levelled
            in place.
        mask: What that masking refused.
        table: The centre wavelength of every column and band, which sets how
            many bands the smoothing window covers.
        detector: Which detector, `l` or `s`, which picks the threshold.

    Returns:
        The mask with each levelled column and band recorded.
    """
    columns, bands = ~mask.columns, ~mask.bands
    live_columns, live_bands = np.flatnonzero(columns), np.flatnonzero(bands)
    # Gathered once rather than an axis at a time, which would copy the cube twice.
    block = cube[np.ix_(np.arange(cube.shape[0]), live_columns, live_bands)]

    # Average each column down the scan, so the ground averages away.
    averaged = block.mean(axis=0)
    # How far each band sits from the median of its wavelength neighbours.
    size = bands_calibration.window(
        bands_calibration.centres(table)[bands], configs.STRIPE_WIDTH
    )
    apart = np.abs(averaged - medfilt1(averaged, size))
    # crism_ml judges each column against the spread of its own bands.
    sigma = configs.STRIPE_SIGMA[detector]
    limit = apart.mean(axis=-1, keepdims=True) + sigma * apart.std(
        ddof=1, axis=-1, keepdims=True
    )
    caught = apart > limit

    # Where each caught cell of the block sits in the cube it was taken from.
    at_column, at_band = np.nonzero(caught)
    if at_column.size:
        # Only a column holding a caught cell is worth smoothing, and some hold none.
        live = np.unique(at_column)
        smoothed = medfilt1(block[:, live], size)
        cube[:, live_columns[at_column], live_bands[at_band]] = smoothed[
            :, np.searchsorted(live, at_column), at_band
        ]

    everywhere = np.zeros(cube.shape[1:], dtype=bool)
    everywhere[np.ix_(columns, bands)] = caught
    return replace(mask, stripes=everywhere)


def medfilt1(array: np.ndarray, size: int, out: np.ndarray | None = None) -> np.ndarray:
    """Return a moving median along the last axis, truncated at the ends.

    Args:
        array: The values to filter.
        size: How many samples wide the window is.
        out: The array to fill, or None to allocate one.

    Returns:
        The filtered values, the same shape as the input.
    """
    left, right = size // 2, size - size // 2
    if out is None:
        out = np.empty_like(array)
    for at in range(array.shape[-1]):
        window = array[..., max(at - left, 0) : at + right]
        held = window.shape[-1]
        # Partitioned rather than sorted, which stops at the middle and writes in place.
        part = np.partition(window, held // 2, axis=-1)
        out[..., at] = (
            part[..., held // 2]
            if held % 2
            else 0.5 * (part[..., held // 2 - 1] + part[..., held // 2])
        )
    return out
