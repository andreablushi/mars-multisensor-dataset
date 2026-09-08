"""Reading one SHARAD track off disk and cutting it to the feature it was kept for."""

from __future__ import annotations

from building.common.pds import images, labels, tables
from building.configs import sharad as configs
from building.preprocessing.common.crop import overlap
from building.preprocessing.sharad import elevation
from building.preprocessing.sharad.models.observation import SharadObservation
from building.preprocessing.sharad.models.sample import SharadSample
from shared.models.feature import Feature

# The field the geometry names each radargram column in, counted from one.
COLUMN_FIELD = "RADARGRAM COLUMN"


def read_observation(identifier: str) -> SharadObservation:
    """Read one radargram and join it to the geometry it was measured at.

    Args:
        identifier: The observation, whose files must already be in the cache
            that `download.fetch` puts them in.

    Returns:
        observation: The observation holding only the traces the geometry places, in the
            radargram's own order.

    Raises:
        FileNotFoundError: When either product or its label is missing.
        KeyError: When a label names a sample type this cannot read.
        ValueError: When the geometry holds fewer rows than its label promises.
    """
    held = {
        kind: configs.CACHE.files(
            identifier, configs.NAMING.product(identifier, kind), kind
        )
        for kind in configs.KINDS
    }
    # The echoes themselves, then the places they were sounded at.
    power, sounding = images.load_plane(held[configs.OBSERVATION][".img"])
    geometry, placing = tables.load_table(held[configs.GEOMETRY][".tab"])
    # The geometry counts columns from one, and the radargram from zero.
    traces = geometry[COLUMN_FIELD].astype("i8") - 1
    return SharadObservation(
        identifier,
        labels.merge(sounding, placing),
        power[:, traces],
        geometry,
        traces,
        elevation.elevation_m(power.shape[0]),
    )


def crop(observation: SharadObservation, frame: Feature) -> SharadSample | None:
    """Return one track holding only the traces its feature's box keeps.

    Args:
        observation: The radargram holding only the traces its geometry places.
        frame: The local frame of the feature it was kept for.

    Returns:
        sample: The track cut to that feature, or None where it reaches none of it.
    """
    held = overlap(
        observation.latitude, observation.longitude, observation.separable, frame
    )
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
        elevation=observation.elevation,
    )
