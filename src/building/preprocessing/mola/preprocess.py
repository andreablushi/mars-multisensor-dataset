"""Reading one MOLA tile off disk and cutting it to the feature it was kept for."""

from __future__ import annotations

from building.common.pds import images, labels
from building.configs import mola as configs
from building.models.feature import FeatureFrame
from building.preprocessing.common.crop import overlap, taken
from building.preprocessing.mola import projection
from building.preprocessing.mola.models.observation import MolaObservation
from building.preprocessing.mola.models.sample import MolaSample


def read_observation(identifier: str) -> MolaObservation:
    """Read one tile onto the grid its own label projects it onto.

    Args:
        identifier: The tile, whose files must already be in the cache that
            `download.fetch` puts them in.

    Returns:
        The observation, its height on the grid its label places it on.

    Raises:
        FileNotFoundError: When the plane or its label is missing.
        KeyError: When the label names a sample type this cannot read.
        ValueError: When it names a projection this cannot read.
    """
    product = configs.NAMING.product(identifier, configs.TOPOGRAPHY)
    height, label = images.load_plane(
        configs.CACHE.files(identifier, product, configs.TOPOGRAPHY)[".img"]
    )
    return MolaObservation(
        identifier, labels.merge(label), height, *projection.load(label)
    )


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
        label=observation.label,
        inside=held.inside,
        topography=taken(observation.topography, held.bounds),
    )
