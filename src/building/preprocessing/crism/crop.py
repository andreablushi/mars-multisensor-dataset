"""Cutting one cleaned CRISM observation to the tiles it was kept for."""

from __future__ import annotations

from building.preprocessing.common import cut, geometry
from building.preprocessing.common.models.samples import Samples
from building.preprocessing.crism.models.observation import (
    ACQUISITION_PLANES,
    CrismObservation,
)
from building.preprocessing.crism.models.sample import CrismSample
from common.models.tile import Tile


def crop(observation: CrismObservation, frame: Tile) -> CrismSample | None:
    """Return one observation holding only the pixels its tile's box keeps.

    Args:
        observation: The observation with its two detectors joined.
        frame: The local frame of the tile it was kept for.

    Returns:
        sample: The observation cut to that tile, or None where it misses.
    """
    # A pushbroom swath bends, so every pixel carries its own backplanes' pair.
    samples = Samples(observation.latitude, observation.longitude, False, None)
    held = cut.overlap(samples, frame)
    if held is None:
        return None
    return CrismSample(
        identifier=observation.identifier,
        position=held.position,
        label=observation.label,
        inside=held.inside,
        valid=geometry.partial_mask(geometry.taken(observation.valid, held.bounds)),
        cube=geometry.taken(observation.cube, held.bounds),
        measured_bands=observation.measured_bands,
        **{
            name: geometry.taken(observation.geometry[:, :, at], held.bounds)
            for name, at in ACQUISITION_PLANES.items()
        },
    )
