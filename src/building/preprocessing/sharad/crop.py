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
    radargram_columns,
)
from building.preprocessing.sharad.models.sample import SharadSample
from common.models.tile import Tile


def trace_position(geometry: np.recarray) -> Position:
    """Return where every trace of one track's geometry was sounded."""
    # A sounder walks a line, so every trace carries its own geometry's pair.
    return Position(geometry[LATITUDE_FIELD], geometry[LONGITUDE_FIELD], False)


def kept_columns(placing: np.recarray, frames: Sequence[Tile]) -> np.ndarray:
    """Return every radargram column the boxes of one track's tiles keep.

    Args:
        placing: The track's geometry, one row per placed trace.
        frames: The local frames of the tiles it is cut to.

    Returns:
        columns: The sorted columns any of them keeps, counted from zero.
    """
    # Cut as `crop` cuts, so exactly the columns it goes on to read are kept.
    position = trace_position(placing)
    traces = radargram_columns(placing)
    held = [cut.overlap(position, frame) for frame in frames]
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
    held = cut.overlap(trace_position(observation.geometry), frame)
    if held is None:
        return None
    # The traces are the radargram's second axis, and the delay is left whole.
    (traces,) = held.bounds
    power = observation.power[:, traces]
    columns = observation.traces[traces]
    placing = observation.geometry[traces]
    return SharadSample(
        identifier=observation.identifier,
        position=held.position,
        label=observation.label,
        inside=held.inside,
        # The archive sounds a trace or fills it whole, so one flag covers its delays.
        valid=np.isfinite(power).all(axis=0),
        power=power,
        clutter=observation.clutter[:, columns],
        traces=columns,
        incidence_deg=placing[SOLAR_ZENITH_FIELD],
        spacecraft_altitude_km=(
            placing[SPACECRAFT_RADIUS_FIELD] - placing[MARS_RADIUS_FIELD]
        ),
    )
