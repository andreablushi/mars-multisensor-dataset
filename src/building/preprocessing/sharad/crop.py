"""Cutting one SHARAD track down to what its feature's box keeps."""

from __future__ import annotations

from building.metadata.models.feature import FeatureFrame
from building.preprocessing.common.crop import overlap
from building.preprocessing.sharad.models.observation import SharadObservation
from building.preprocessing.sharad.models.sample import SharadSample


def crop(observation: SharadObservation, frame: FeatureFrame) -> SharadSample | None:
    """Return one track holding only the traces its feature's box keeps.

    Args:
        observation: The radargram holding only the traces its geometry places.
        frame: The local frame of the feature it was kept for.

    Returns:
        The track cut to that feature, or None where it reaches none of it.
    """
    held = overlap(observation, frame)
    if held is None:
        return None
    # A sounder walks a line, so the traces are the radargram's second axis and
    # the delay each one was sounded over is left whole.
    (traces,) = held.bounds
    return SharadSample(
        identifier=observation.identifier,
        position=held.position,
        inside=held.inside,
        power=observation.power[:, traces],
        geometry=observation.geometry[traces],
        traces=observation.traces[traces],
    )
