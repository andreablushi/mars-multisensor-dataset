"""Reading one SHARAD track off disk and cutting it to the feature it was kept for."""

from __future__ import annotations

from building.common.pds import images, labels, tables
from building.configs import sharad as configs
from building.models.feature import FeatureFrame
from building.preprocessing.common.crop import overlap
from building.preprocessing.sharad.models.observation import SharadObservation
from building.preprocessing.sharad.models.sample import SharadSample

# The field the geometry names each radargram column in, counted from one.
COLUMN_FIELD = "RADARGRAM COLUMN"


def read_observation(identifier: str) -> SharadObservation:
    """Read one radargram and join it to the geometry it was measured at.

    Args:
        identifier: The observation, whose files must already be in the cache
            that `download.fetch` puts them in.

    Returns:
        The observation holding only the traces the geometry places, in the
        order the radargram stores them.

    Raises:
        FileNotFoundError: When either product or its label is missing.
        KeyError: When a label names a sample type this cannot read.
        ValueError: When the geometry holds fewer rows than its label promises.
    """
    # The echoes themselves, then the places they were sounded at.
    power, sounding = images.load_plane(
        configs.CACHE.files(
            identifier,
            configs.NAMING.product(identifier, configs.OBSERVATION),
            configs.OBSERVATION,
        )[".img"]
    )
    geometry, placing = tables.load_table(
        configs.CACHE.files(
            identifier,
            configs.NAMING.product(identifier, configs.GEOMETRY),
            configs.GEOMETRY,
        )[".tab"]
    )
    # The geometry counts columns from one, and the radargram from zero.
    traces = geometry[COLUMN_FIELD].astype("i8") - 1
    return SharadObservation(
        identifier,
        labels.merge(sounding, placing),
        power[:, traces],
        geometry,
        traces,
    )


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
    # The traces are the radargram's second axis, and the delay is left whole.
    (traces,) = held.bounds
    return SharadSample(
        identifier=observation.identifier,
        position=held.position,
        label=observation.label,
        inside=held.inside,
        power=observation.power[:, traces],
        geometry=observation.geometry[traces],
        traces=observation.traces[traces],
    )
