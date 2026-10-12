"""Resampling one MOLA grid onto square ground samples around a tile centre."""

from __future__ import annotations

import math

import numpy as np

from building.configs import mola as configs
from building.preprocessing.common import polar
from building.preprocessing.common.models.position import Position
from common.maths import geodesy
from common.maths.geodesy import SPHEROID, TURN
from common.models.tile import Tile


def square_position(frame: Tile) -> Position:
    """Return the square samples covering one tile's box, in absolute degrees.

    Args:
        frame: The local frame of the tile, whose centre they are walked from.

    Returns:
        position: The latitude and longitude of every sample, lines falling southward.
    """
    lon, lat = geodesy.bbox_ring(frame, polar.STEP)
    azimuth, _, span = SPHEROID.inv(
        np.full(lon.shape, frame.centre_lon),
        np.full(lat.shape, frame.centre_lat),
        lon,
        lat,
    )
    bearing = np.radians(azimuth)
    up, across = (
        math.ceil(np.abs(span * turned(bearing)).max() / configs.SPACING_M)
        for turned in (np.cos, np.sin)
    )
    north, east = np.meshgrid(
        np.arange(up, -up - 1, -1) * configs.SPACING_M,
        np.arange(-across, across + 1) * configs.SPACING_M,
        indexing="ij",
    )
    lon, lat, _ = SPHEROID.fwd(
        np.full(north.shape, frame.centre_lon),
        np.full(north.shape, frame.centre_lat),
        np.degrees(np.arctan2(east, north)),
        np.hypot(east, north),
    )
    return Position(lat, geodesy.normalise_longitude(lon), False)


def resampled_elevation(
    elevation: np.ndarray, source: Position, target: Position
) -> tuple[np.ndarray, np.ndarray]:
    """Return one grid's elevation read bilinearly at other samples.

    Args:
        elevation: The grid's elevation in metres, lines by samples.
        source: Where its lines and samples sit, evenly spaced in degrees or metres.
        target: The samples to read it at, in absolute degrees.

    Returns:
        elevation: The elevation at every target in whole metres, zero where unread.
        covered: Which targets fall between four of the grid's bins.
    """
    lon, lat = target.east, target.north
    if source.grid is None:
        along = (lat, source.east[0] + (lon - source.east[0]) % TURN)
    else:
        along = geodesy.stereographic_forward(lon, lat, *source.grid)[::-1]
    line, sample = (
        (at - axis[0]) / (axis[1] - axis[0])
        for at, axis in zip(along, (source.north, source.east), strict=True)
    )
    lines, samples = elevation.shape
    covered = (
        (line >= 0) & (line <= lines - 1) & (sample >= 0) & (sample <= samples - 1)
    )
    top = np.clip(np.floor(line), 0, lines - 2).astype(int)
    left = np.clip(np.floor(sample), 0, samples - 2).astype(int)
    down = np.clip(line - top, 0.0, 1.0)
    across = np.clip(sample - left, 0.0, 1.0)
    held = elevation.astype(float)
    upper = held[top, left] * (1.0 - across) + held[top, left + 1] * across
    lower = held[top + 1, left] * (1.0 - across) + held[top + 1, left + 1] * across
    read = upper * (1.0 - down) + lower * down
    return np.where(covered, np.rint(read), 0.0).astype(elevation.dtype), covered
