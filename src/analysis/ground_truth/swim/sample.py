"""Every tile between 60S and 60N, sampled on one grid under the SWIM maps."""

from __future__ import annotations

import numpy as np

from analysis.ground_truth.swim.maps import (
    LAYERS,
    NORTH_LAT,
    map_pixels,
    read_consistency_map,
)
from analysis.ground_truth.swim.models import IceLabel, SwimSettings
from analysis.partition import Tessellate
from common.maths.geodesy import SPHEROID
from common.maths.physics import METRES_PER_KM

ICY_SHARE = 0.5


def grid_points(
    centre_lon: np.ndarray, centre_lat: np.ndarray, settings: SwimSettings
) -> tuple[np.ndarray, np.ndarray]:
    """Return where every tile's grid points stand, rows from the north.

    Args:
        centre_lon: The longitude each tile is centred on, in degrees. (T,)
        centre_lat: The latitude each tile is centred on, in degrees. (T,)
        settings: The settled choices, sizing the grid.

    Returns:
        lon: The longitude of every point, -180 to 180. (T, P * P)
        lat: The latitude of every point. (T, P * P)
    """
    steps = (np.arange(settings.points) - (settings.points - 1) / 2) * (
        settings.spacing_km * METRES_PER_KM
    )
    north, east = np.meshgrid(steps[::-1], steps, indexing="ij")
    # Placed by the same geodesic east and north of the centre the tokens are
    azimuth = np.degrees(np.arctan2(east, north)).ravel()
    span = np.hypot(east, north).ravel()
    count = len(centre_lon)
    lon, lat, _ = SPHEROID.fwd(
        np.repeat(centre_lon, azimuth.size),
        np.repeat(centre_lat, azimuth.size),
        np.tile(azimuth, count),
        np.tile(span, count),
    )
    return lon.reshape(count, -1), lat.reshape(count, -1)


def ice_labels(grid: Tessellate, settings: SwimSettings) -> list[IceLabel]:
    """Label every tile the SWIM maps cover whole with the ice under its grid.

    Args:
        grid: The grid Mars is split into.
        settings: The settled choices for the sampling.

    Returns:
        labels: One label per tile between 60S and 60N, south to north.
    """
    mapped = [
        tile
        for band, count in enumerate(grid.columns)
        for tile in (grid.tile_of(band, column) for column in range(count))
        if -NORTH_LAT <= tile.min_lat and tile.max_lat <= NORTH_LAT
    ]
    centres = np.array([(tile.centre_lon, tile.centre_lat) for tile in mapped])
    pixels = map_pixels(*grid_points(centres[:, 0], centres[:, 1], settings))
    shallow, deep = (read_consistency_map(layer)[pixels] for layer in LAYERS)
    icy_shallow = np.mean(shallow >= settings.threshold_1_5m, axis=1) >= ICY_SHARE
    icy_deep = np.mean(deep >= settings.threshold_5m, axis=1) >= ICY_SHARE
    return [
        IceLabel(
            tile=tile.name,
            centre_lon=float(lon),
            centre_lat=float(lat),
            consistency_1_5m=tuple(shallow[at].tolist()),
            consistency_5m=tuple(deep[at].tolist()),
            ice_1_5m=bool(icy_shallow[at]),
            ice_5m=bool(icy_deep[at]),
        )
        for at, (tile, (lon, lat)) in enumerate(zip(mapped, centres, strict=True))
    ]
