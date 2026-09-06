"""Cutting one CTX scan down to what its feature's box keeps."""

from __future__ import annotations

from building.metadata.models.feature import FeatureFrame
from building.preprocessing.common.crop import overlap, taken
from building.preprocessing.ctx.models.observation import CtxObservation
from building.preprocessing.ctx.models.sample import CtxSample


def crop(observation: CtxObservation, frame: FeatureFrame) -> CtxSample | None:
    """Return one scan holding only the pixels its feature's box keeps.

    Args:
        observation: The scan as it was read off disk.
        frame: The local frame of the feature it was kept for.

    Returns:
        The scan cut to that feature, or None where it reaches none of it.
    """
    held = overlap(observation, frame)
    if held is None:
        return None
    return CtxSample(
        identifier=observation.identifier,
        position=held.position,
        inside=held.inside,
        image=taken(observation.image, held.bounds),
    )
