"""Dividing each spectrum by a bland one from its own column."""

from __future__ import annotations

import numpy as np


def ratio_colmed(pixspec: np.ndarray, rem: np.ndarray) -> None:
    """Use the median of a column for ratioing, as crism_ml's ColMed does.

    Args:
        pixspec: The values as lines by samples by bands, divided through in
            place.
        rem: Lines by samples, True where the pixel is not a measurement and so
            is kept out of the median.

    Returns:
        None.
    """
    for at in range(pixspec.shape[1]):
        live = ~rem[:, at]
        # A column with no measurement has nothing to ratio, and is refused anyway.
        if live.any():
            column = pixspec[:, at, :]
            held = column[live]
            column[live] = held / np.median(held, axis=0)
    pixspec[rem] = 0.0
