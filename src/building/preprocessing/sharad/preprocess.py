"""Reading one SHARAD track off disk and cutting it to the tile it was kept for."""

from __future__ import annotations

import numpy as np

from building.common.pds import images, labels, tables
from building.configs import sharad as configs
from building.preprocessing.common import geometry
from building.preprocessing.common.models.samples import Samples
from building.preprocessing.sharad.models.observation import SharadObservation
from building.preprocessing.sharad.models.sample import SharadSample
from shared.models.tile import Tile

# The field the geometry names each radargram column in, counted from one.
COLUMN_FIELD = "RADARGRAM COLUMN"


def read_observation(identifier: str) -> SharadObservation:
    """Read one radargram and join it to the geometry it was measured at.

    Args:
        identifier: The observation, whose files must already be in the cache
            that `download.fetch` puts them in.

    Returns:
        observation: The observation holding only the traces the geometry places, in the
            radargram's own order, with the combined clutter simulation beside them.

    Raises:
        FileNotFoundError: When any product or a label is missing.
        KeyError: When a label names a sample type this cannot read.
        ValueError: When the geometry holds fewer rows than its label promises, or
            the clutter is not one array the radargram's size.
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
    simulated = held[configs.CLUTTER][".img"]
    if simulated.stat().st_size != power.size * np.dtype(configs.CLUTTER_TYPE).itemsize:
        raise ValueError(f"{simulated.name} is not one array the radargram's size.")
    return SharadObservation(
        identifier,
        labels.merge(sounding, placing),
        power[:, traces],
        np.memmap(simulated, dtype=configs.CLUTTER_TYPE, mode="r", shape=power.shape),
        geometry,
        traces,
    )


def crop(observation: SharadObservation, frame: Tile) -> SharadSample | None:
    """Return one track holding only the traces its tile's box keeps.

    Args:
        observation: The radargram holding only the traces its geometry places.
        frame: The local frame of the tile it was kept for.

    Returns:
        sample: The track cut to that tile, or None where it reaches none of it.
    """
    held = geometry.overlap(
        Samples(
            observation.latitude, observation.longitude, observation.separable, None
        ),
        frame,
    )
    if held is None:
        return None
    # The traces are the radargram's second axis, and the delay is left whole.
    (traces,) = held.bounds
    power = observation.power[:, traces]
    return SharadSample(
        identifier=observation.identifier,
        position=held.position,
        label=observation.label,
        inside=held.inside,
        # The archive sounds a trace or fills it whole, so one flag covers its delays.
        valid=np.isfinite(power).all(axis=0),
        power=power,
        clutter=observation.clutter[:, observation.traces[traces]],
        traces=observation.traces[traces],
    )
