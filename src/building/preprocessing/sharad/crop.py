"""Cutting one SHARAD track to the tiles it was kept for."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from building.preprocessing.common import cut
from building.preprocessing.common.models.position import Position
from building.preprocessing.sharad.models.observation import (
    LATITUDE_FIELD,
    LONGITUDE_FIELD,
    MARS_RADIUS_FIELD,
    SOLAR_ZENITH_FIELD,
    SPACECRAFT_RADIUS_FIELD,
    SharadObservation,
)
from building.preprocessing.sharad.models.sample import SharadSample
from common.models.tile import Tile

FILL = 0.0


def trace_position(geometry: np.recarray) -> Position:
    """Return where every trace of one track's geometry was sounded."""
    # A sounder walks a line, so every trace carries its own geometry's pair.
    return Position(geometry[LATITUDE_FIELD], geometry[LONGITUDE_FIELD], False)


def kept_columns(geometry: np.recarray, frames: Sequence[Tile]) -> np.ndarray:
    """Return every radargram column the boxes of one track's tiles keep.

    Args:
        geometry: The track's geometry, one row per placed trace.
        frames: The local frames of the tiles it is cut to.

    Returns:
        columns: The sorted columns any of them keeps, counted from zero.
    """
    # Cut as `crop` cuts, so exactly the columns it goes on to read are kept.
    position = trace_position(geometry)
    held = [cut.overlap(position, frame) for frame in frames]
    return np.unique(
        np.concatenate(
            [one.bounds[0] for one in held if one is not None] + [np.empty(0, "i8")]
        )
    )


def crop(observation: SharadObservation, frame: Tile) -> SharadSample | None:
    """Return one track holding only the traces its tile's box keeps.

    Args:
        observation: The radargram with its geometry joined onto it.
        frame: The local frame of the tile it was kept for.

    Returns:
        sample: The track cut to that tile, or None where it reaches none of it.
    """
    held = cut.overlap(trace_position(observation.geometry), frame)
    if held is None:
        return None
    # The traces are the radargram's second axis, and the delay is left whole.
    (traces,) = held.bounds
    power = observation.power[:, traces]
    # The archive sounds a trace or fills it whole, so one flag covers its delays.
    valid = np.isfinite(power).all(axis=0)
    geometry = observation.geometry[traces]
    return SharadSample(
        position=held.position,
        label=observation.label,
        inside=held.inside,
        valid=valid,
        power=np.where(valid, power, FILL),
        clutter=observation.clutter[:, traces],
        traces=traces,
        incidence_deg=geometry[SOLAR_ZENITH_FIELD],
        spacecraft_altitude_km=(
            geometry[SPACECRAFT_RADIUS_FIELD] - geometry[MARS_RADIUS_FIELD]
        ),
    )
