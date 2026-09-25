"""Reading one SHARAD track off disk and cutting it to the tile it was kept for."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from building.configs import sharad as configs
from building.preprocessing.common import geometry
from building.preprocessing.common.models.samples import Samples
from building.preprocessing.sharad.models.observation import (
    LATITUDE_FIELD,
    LONGITUDE_FIELD,
    SharadObservation,
)
from building.preprocessing.sharad.models.sample import SharadSample
from common.models.tile import Tile
from common.pds import images, labels, tables

# The field the geometry names each radargram column in, counted from one.
COLUMN_FIELD = "RADARGRAM COLUMN"


def kept_columns(placing: np.recarray, frames: Sequence[Tile]) -> np.ndarray:
    """Return every radargram column the boxes of one track's tiles keep.

    Args:
        placing: The track's geometry, one row per placed trace.
        frames: The local frames of the tiles it is cut to.

    Returns:
        columns: The sorted columns any of them keeps, counted from zero.
    """
    # Cut as `crop` cuts, so exactly the columns it goes on to read are kept.
    samples = Samples(
        placing[LATITUDE_FIELD],
        placing[LONGITUDE_FIELD],
        SharadObservation.separable,
        None,
    )
    traces = placing[COLUMN_FIELD].astype("i8") - 1
    held = [geometry.overlap(samples, frame) for frame in frames]
    return np.unique(
        np.concatenate(
            [traces[one.bounds[0]] for one in held if one is not None]
            + [np.empty(0, "i8")]
        )
    )


def read_observation(identifier: str) -> SharadObservation:
    """Read one radargram and join it to the geometry it was measured at.

    Args:
        identifier: The observation, its files already in the download cache.

    Returns:
        observation: The placed traces in order, with their clutter simulation.

    Raises:
        FileNotFoundError: When any product or a label is missing.
        KeyError: When a label names a sample type this cannot read.
        ValueError: When the geometry is short or the clutter does not fit.
    """
    held = {
        kind: configs.CACHE.files(
            identifier, configs.NAMING.product(identifier, kind), kind
        )
        for kind in configs.Kind
    }
    # The echoes themselves, then the places they were sounded at.
    power, sounding = images.load_cube(held[configs.Kind.OBSERVATION][".img"])
    power = power[:, :, 0]
    geometry, placing = tables.load_table(held[configs.Kind.GEOMETRY][".tab"])
    # The geometry counts columns from one, and the radargram from zero.
    traces = geometry[COLUMN_FIELD].astype("i8") - 1
    simulated = held[configs.Kind.CLUTTER][".img"]
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
        incidence_deg=geometry.taken(observation.solar_zenith_deg, held.bounds),
        spacecraft_altitude_km=geometry.taken(
            observation.spacecraft_altitude_km, held.bounds
        ),
    )
