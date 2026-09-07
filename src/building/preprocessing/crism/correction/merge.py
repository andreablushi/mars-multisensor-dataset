"""Joining the detectors one observation was delivered as, and its geometry."""

from __future__ import annotations

import numpy as np

from building.preprocessing.crism.correction import bands_calibration
from building.preprocessing.crism.models.detector import Detector
from building.preprocessing.crism.models.observation import CrismObservation

# Which detector carries which half, and the order their bands are laid out in.
VISIBLE = "s"
INFRARED = "l"
HALVES = (VISIBLE, INFRARED)


def merge_detectors(
    identifier: str,
    detectors: dict[str, Detector],
    geometry: np.ndarray,
    label: dict[str, str],
) -> CrismObservation:
    """Join the detectors of a cleaned observation into one cube.

    The two are read out together, so they share a grid, but one of them can
    lose the last frames of a strip the other kept and a small share of the
    survey was archived as a single half. Both are met by taking the lines every
    half and the geometry all carry, and by joining whichever halves landed.

    Args:
        identifier: The observation the detectors are halves of.
        detectors: The halves that landed, already through
            `preprocess.clean_detectors`, so each carries the mask saying what
            it kept.
        geometry: The backplanes that place every pixel, on the same grid.
        label: What every product the observation was published as says of it.

    Returns:
        The joined observation, its bands ascending in wavelength.

    Raises:
        ValueError: When no half was delivered, or one has not been cleaned.
    """
    halves = tuple(name for name in HALVES if name in detectors)
    if not halves:
        raise ValueError(f"{identifier} was delivered as no detector.")
    if any(detectors[name].mask is None for name in halves):
        raise ValueError(f"{identifier} has not been cleaned.")

    # Every half is read out from the first frame, so the shortest ends the strip.
    lines = min(geometry.shape[0], *(detectors[name].cube.shape[0] for name in halves))
    # Only the samples no half refused.
    columns = ~np.logical_or.reduce([detectors[name].mask.columns for name in halves])

    kept = np.flatnonzero(columns)
    bands = {name: ~detectors[name].mask.bands for name in halves}
    table = np.concatenate(
        [detectors[name].wavelengths[columns][:, bands[name]] for name in halves],
        axis=1,
    )
    # The two overlap around a micron, so ordering is a sort and not a join.
    order = np.argsort(bands_calibration.centres(table))
    # Where each band lands once ordered, so each half writes straight into the cube.
    lands = np.empty(order.size, dtype="i8")
    lands[order] = np.arange(order.size)

    joined = np.empty((lines, kept.size, order.size), dtype="f4")
    at = 0
    for name in halves:
        live = np.flatnonzero(bands[name])
        joined[:, :, lands[at : at + live.size]] = detectors[name].cube[
            np.ix_(np.arange(lines), kept, live)
        ]
        at += live.size
    return CrismObservation(
        identifier,
        label,
        joined,
        table[:, order],
        geometry[:lines, columns],
        kept,
        # A pixel any half could not read is no measurement of the observation.
        ~np.logical_or.reduce([detectors[name].mask.pixels[:lines] for name in halves])[
            :, columns
        ],
    )
