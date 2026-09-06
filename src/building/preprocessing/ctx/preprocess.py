"""Reading one CTX scan off disk and cutting it to the feature it was kept for."""

from __future__ import annotations

import tifffile

from building.common.pds import labels
from building.configs import ctx as configs
from building.models.feature import FeatureFrame
from building.preprocessing.common.crop import marked, overlap, taken
from building.preprocessing.ctx import projection
from building.preprocessing.ctx.models.observation import CtxObservation
from building.preprocessing.ctx.models.sample import BLANK, CtxSample


def read_observation(identifier: str) -> CtxObservation:
    """Read one scan and place it on the grid its label projects it onto.

    Args:
        identifier: The observation, whose files must already be in the cache
            that `download.fetch` puts them in.

    Returns:
        The observation, its image on that grid.

    Raises:
        FileNotFoundError: When the image or its label is missing.
        ValueError: When the label names a projection this cannot read, or the
            image holds more than one plane.
    """
    files = configs.CACHE.files(identifier, identifier)
    label = labels.load(files[configs.SUFFIXES[configs.LABEL]])
    # ASU publishes the pixels as a TIFF rather than beside a label of their own.
    image = tifffile.imread(files[configs.SUFFIXES[configs.IMAGE]])
    if image.ndim != 2:
        raise ValueError(f"{identifier} holds a {image.ndim} dimensional image.")
    return CtxObservation(
        identifier, labels.merge(label), image, *projection.load(label)
    )


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
    image = taken(observation.image, held.bounds)
    return CtxSample(
        identifier=observation.identifier,
        position=held.position,
        label=observation.label,
        inside=held.inside,
        valid=marked(image != BLANK),
        image=image,
    )
