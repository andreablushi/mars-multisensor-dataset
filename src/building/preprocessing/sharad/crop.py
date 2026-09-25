"""Cutting one SHARAD track to the tiles it was kept for."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from building.preprocessing.common import cut, geometry
from building.preprocessing.common.models.samples import Samples
from building.preprocessing.sharad.models.observation import (
    COLUMN_FIELD,
    LATITUDE_FIELD,
    LONGITUDE_FIELD,
    SharadObservation,
)
from building.preprocessing.sharad.models.sample import SharadSample
from common.models.tile import Tile


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
    held = [cut.overlap(samples, frame) for frame in frames]
    return np.unique(
        np.concatenate(
            [traces[one.bounds[0]] for one in held if one is not None]
            + [np.empty(0, "i8")]
        )
    )


def crop(observation: SharadObservation, frame: Tile) -> SharadSample | None:
    """Return one track holding only the traces its tile's box keeps.

    Args:
        observation: The radargram holding only the traces its geometry places.
        frame: The local frame of the tile it was kept for.

    Returns:
        sample: The track cut to that tile, or None where it reaches none of it.
    """
    held = cut.overlap(
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
