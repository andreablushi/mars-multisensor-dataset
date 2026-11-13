"""Cutting one SHARAD track to the tiles it was kept for."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from mars_multisensor_dataset.building.configs import sharad as configs
from mars_multisensor_dataset.building.preprocessing.common import cut
from mars_multisensor_dataset.building.preprocessing.common.models.position import (
    Position,
)
from mars_multisensor_dataset.building.preprocessing.sharad.denoise import (
    denoised,
    measured_noise,
)
from mars_multisensor_dataset.building.preprocessing.sharad.models.observation import (
    LATITUDE_FIELD,
    LONGITUDE_FIELD,
    MARS_RADIUS_FIELD,
    SOLAR_ZENITH_FIELD,
    SPACECRAFT_RADIUS_FIELD,
    SharadObservation,
)
from mars_multisensor_dataset.building.preprocessing.sharad.models.sample import (
    SharadSample,
)
from mars_multisensor_dataset.building.preprocessing.sharad.normalize import (
    mean_surface_peak,
)
from mars_multisensor_dataset.building.preprocessing.sharad.window import (
    FEWEST_ROWS,
    rows_below_atmosphere,
)
from mars_multisensor_dataset.common.models.tile import Tile

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
        sample: The track cut to that tile, the radargram gated at its noise
            threshold and the clutter as simulated, or None where it measures
            none of it, holds no echo, or keeps under `FEWEST_ROWS` rows below
            its atmosphere.
    """
    held = cut.overlap(trace_position(observation.geometry), frame)
    if held is None:
        return None
    # The traces are the radargram's second axis, and the delay is left whole.
    (traces,) = held.bounds
    power = observation.power[:, traces]
    # The archive sounds a trace or fills it whole, so one flag covers its delays.
    valid = np.isfinite(power).all(axis=0)
    measured = valid if held.inside is None else valid & held.inside
    if not measured.any():
        return None
    noise = measured_noise(power[:, valid], observation.gate)
    if noise is None:
        return None
    gated = np.full(power.shape, FILL, power.dtype)
    gated[:, valid] = denoised(noise.averaged, noise.threshold, noise.spread)
    below = rows_below_atmosphere(gated.astype(configs.LAYOUT.stored), measured)
    if below.stop - below.start < FEWEST_ROWS:
        return None
    geometry = observation.geometry[traces]
    return SharadSample(
        position=held.position,
        label=observation.label,
        inside=held.inside,
        valid=valid,
        power=gated,
        clutter=observation.clutter[:, traces],
        traces=traces,
        incidence_deg=geometry[SOLAR_ZENITH_FIELD],
        spacecraft_altitude_km=(
            geometry[SPACECRAFT_RADIUS_FIELD] - geometry[MARS_RADIUS_FIELD]
        ),
        noise_threshold_db=noise.threshold,
        noise_spread_db=noise.spread,
        surface_offset_db=mean_surface_peak(power, measured),
    )
