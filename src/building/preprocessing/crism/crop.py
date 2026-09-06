"""Cutting one CRISM observation down to what its feature's box keeps."""

from __future__ import annotations

from building.metadata.models.feature import FeatureFrame
from building.preprocessing.common.crop import overlap, taken
from building.preprocessing.crism.models.observation import CrismObservation
from building.preprocessing.crism.models.sample import CrismSample


def crop(observation: CrismObservation, frame: FeatureFrame) -> CrismSample | None:
    """Return one observation holding only the pixels its feature's box keeps.

    Args:
        observation: The observation with its two detectors joined.
        frame: The local frame of the feature it was kept for.

    Returns:
        The observation cut to that feature, its bands left whole, or None
        where it reaches none of it.
    """
    held = overlap(observation, frame)
    if held is None:
        return None
    # The detector is calibrated column by column, so the wavelengths and the
    # columns are cut by the column axis alone and the bands are left whole.
    columns = held.bounds[1]
    return CrismSample(
        identifier=observation.identifier,
        position=held.position,
        inside=held.inside,
        cube=taken(observation.cube, held.bounds),
        wavelengths=observation.wavelengths[columns],
        columns=observation.columns[columns],
    )
