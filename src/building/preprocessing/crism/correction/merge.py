"""Joining the visible and infrared halves, and the geometry beside them."""

from __future__ import annotations

import numpy as np

from building.preprocessing.crism.correction import bands_calibration
from building.preprocessing.crism.models.detector import Detector
from building.preprocessing.crism.models.observation import CrismObservation

# Which detector carries which half of the spectrum, and the order the two are
# read in, which is the order their bands are laid out before being sorted.
VISIBLE = "s"
INFRARED = "l"
HALVES = (VISIBLE, INFRARED)


def merge_detectors(
    identifier: str, detectors: dict[str, Detector], geometry: np.ndarray
) -> CrismObservation:
    """Join both detectors of a cleaned observation into one cube.

    Args:
        identifier: The observation the two detectors are halves of.
        detectors: Both halves, already through `preprocess.clean_detectors`,
            so each carries the mask saying what it kept.
        geometry: The backplanes that place every pixel, on the same grid.

    Returns:
        The joined observation, its bands ascending in wavelength.

    Raises:
        ValueError: When the observation has not been cleaned.
    """
    visible, infrared = detectors[VISIBLE], detectors[INFRARED]
    if visible.mask is None or infrared.mask is None:
        raise ValueError(f"{identifier} has not been cleaned.")

    # Only the samples neither detector refused, which is one unbroken run.
    columns = ~(visible.mask.columns | infrared.mask.columns)

    kept = np.flatnonzero(columns)
    bands = {name: ~held.mask.bands for name, held in detectors.items()}
    table = np.concatenate(
        [detectors[name].wavelengths[columns][:, bands[name]] for name in HALVES],
        axis=1,
    )
    # The two overlap around a micron, so ordering is a sort and not a join.
    order = np.argsort(bands_calibration.centres(table))
    # Where each band of each half lands once they are ordered, so every half
    # is written straight into the joined cube and none is joined then sorted.
    lands = np.empty(order.size, dtype="i8")
    lands[order] = np.arange(order.size)

    joined = np.empty((*visible.cube.shape[:1], kept.size, order.size), dtype="f4")
    at = 0
    for name in HALVES:
        held = detectors[name]
        live = np.flatnonzero(bands[name])
        joined[:, :, lands[at : at + live.size]] = held.cube[
            np.ix_(np.arange(held.cube.shape[0]), kept, live)
        ]
        at += live.size
    return CrismObservation(
        identifier,
        joined,
        table[:, order],
        geometry[:, columns],
        kept,
    )
