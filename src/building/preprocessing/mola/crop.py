"""Cutting one MOLA tile down to what its feature's box keeps."""

from __future__ import annotations

from building.metadata.models.feature import FeatureFrame
from building.preprocessing.common.crop import overlap, taken
from building.preprocessing.mola.models.observation import MolaObservation
from building.preprocessing.mola.models.sample import MolaSample


def crop(observation: MolaObservation, frame: FeatureFrame) -> MolaSample | None:
    """Return one tile holding only the bins its feature's box keeps.

    Args:
        observation: The tile as it was read off disk.
        frame: The local frame of the feature it was kept for.

    Returns:
        The tile cut to that feature, or None where it reaches none of it.
    """
    held = overlap(observation, frame)
    if held is None:
        return None
    return MolaSample(
        identifier=observation.identifier,
        position=held.position,
        inside=held.inside,
        topography=taken(observation.topography, held.bounds),
        counts=taken(observation.counts, held.bounds),
    )
