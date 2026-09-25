"""crism_ml's per-column spike removal, with thresholds set for survey bands."""

from __future__ import annotations

import numpy as np

from building.configs.crism import STRIPE_SIGMA, Detector
from building.preprocessing.crism.correction import moving_median
from building.preprocessing.crism.models.mask import Mask

# How wide the moving median reaches, in nm so every configuration means the same.
STRIPE_WIDTH = 80.0


def remove_spike_columns(
    cube: np.ndarray, mask: Mask, centre: np.ndarray, detector: Detector
) -> None:
    """Replace every band of a column that spikes away from its neighbours.

    Args:
        cube: The masked values as lines by samples by bands, levelled in place.
        mask: What that masking refused.
        centre: The centre wavelength of every band.
        detector: Which detector, `l` or `s`, which picks the threshold.

    Raises:
        ValueError: When the threshold is beyond what a column's bands can reach.
    """
    live_columns = np.flatnonzero(~mask.columns)
    live_bands = np.flatnonzero(~mask.bands)
    # Gathered once rather than an axis at a time, which would copy the cube twice.
    block = cube[np.ix_(np.arange(cube.shape[0]), live_columns, live_bands)]

    # Average each column down the scan, so the ground averages away.
    averaged = block.mean(axis=0)
    # How far each band sits from the median of its wavelength neighbours.
    size = moving_median.window_size(centre[live_bands], STRIPE_WIDTH)
    apart = np.abs(averaged - moving_median.moving_median(averaged, size))
    # crism_ml judges each column against the spread of its own bands.
    sigma = STRIPE_SIGMA[detector]
    # One of n bands sits at most (n-1)/sqrt(n) off the mean; past that catches none
    reach = (live_bands.size - 1) / np.sqrt(live_bands.size)
    if sigma >= reach:
        raise ValueError(
            f"A {detector} spike at {sigma} deviations cannot be reached by "
            f"{live_bands.size} bands, which reach {reach:.2f}."
        )
    limit = apart.mean(axis=-1, keepdims=True) + sigma * apart.std(
        ddof=1, axis=-1, keepdims=True
    )
    caught = apart > limit

    # Where each caught cell of the block sits in the cube it was taken from.
    at_column, at_band = np.nonzero(caught)
    if at_column.size:
        # Only a column holding a caught cell is worth smoothing, and some hold none.
        live = np.unique(at_column)
        smoothed = moving_median.moving_median(block[:, live], size)
        cube[:, live_columns[at_column], live_bands[at_band]] = smoothed[
            :, np.searchsorted(live, at_column), at_band
        ]
