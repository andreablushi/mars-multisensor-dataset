"""Splitting Mars into equal-area tiles, and gathering them into groups."""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from shared.maths.geodesy import TURN
from shared.maths.physics import RADIUS_M
from shared.models.tile import Tile
from shared.models.tile_group import TileGroup

HALF_TURN = 180.0


@lru_cache(maxsize=4)
def band_columns(tile_km: float) -> tuple[int, ...]:
    """Return how many tiles each latitude band is split into.

    Args:
        tile_km: The side a tile is sized to, in kilometres.

    Returns:
        columns: One count per band, from the south pole north, each band holding
            as many tiles as its area has room for.
    """
    radius_km = RADIUS_M / 1000.0
    bands = max(1, round(math.pi * radius_km / tile_km))
    edges = np.radians(np.linspace(-90.0, 90.0, bands + 1))
    areas = 2.0 * math.pi * radius_km**2 * np.diff(np.sin(edges))
    return tuple(max(1, round(area / tile_km**2)) for area in areas)


def tile_of(band: int, column: int, tile_km: float) -> Tile:
    """Return one tile from where it sits on the grid.

    Args:
        band: The latitude band, counted from the south pole.
        column: The place along that band, counted east from the prime meridian.
        tile_km: The side a tile is sized to, in kilometres.

    Returns:
        tile: The tile, a single cap circling the pole where its band holds one.
    """
    columns = band_columns(tile_km)
    height = HALF_TURN / len(columns)
    width = TURN / columns[band]
    circling = columns[band] == 1
    return Tile(
        band=band,
        column=column,
        min_lat=-90.0 + band * height,
        max_lat=-90.0 + (band + 1) * height,
        west_lon=0.0 if circling else column * width,
        east_lon=0.0 if circling else (column + 1) * width,
    )


def every_tile(tile_km: float) -> list[Tile]:
    """Return every tile Mars is split into.

    Args:
        tile_km: The side a tile is sized to, in kilometres.

    Returns:
        tiles: Every tile, band by band from the south pole and west to east.
    """
    return [
        tile_of(band, column, tile_km)
        for band, count in enumerate(band_columns(tile_km))
        for column in range(count)
    ]


def tile_named(name: str, tile_km: float) -> Tile:
    """Return the tile a name spells.

    Args:
        name: The tile's name, such as "b123_c0456".
        tile_km: The side a tile is sized to, in kilometres.

    Returns:
        tile: The tile it names.
    """
    band, column = name.split("_")
    return tile_of(int(band[1:]), int(column[1:]), tile_km)


def tile_indices(
    lat: np.ndarray | float, lon: np.ndarray | float, tile_km: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return the tile every point falls in.

    Args:
        lat: The latitudes in degrees.
        lon: The longitudes in degrees, any turn.
        tile_km: The side a tile is sized to, in kilometres.

    Returns:
        bands: The band each point falls in.
        columns: The column each point falls in along its band.
    """
    columns = np.asarray(band_columns(tile_km))
    bands = np.clip(
        np.floor((np.asarray(lat, dtype=float) + 90.0) / HALF_TURN * columns.size),
        0,
        columns.size - 1,
    ).astype(np.int64)
    held = columns[bands]
    along = np.floor(np.mod(np.asarray(lon, dtype=float), TURN) / TURN * held)
    return bands, np.minimum(along.astype(np.int64), held - 1)


def tile_at(lat: float, lon: float, tile_km: float) -> Tile:
    """Return the tile one point falls in.

    Args:
        lat: The latitude in degrees.
        lon: The longitude in degrees, any turn.
        tile_km: The side a tile is sized to, in kilometres.

    Returns:
        tile: The tile holding it.
    """
    band, column = tile_indices(lat, lon, tile_km)
    return tile_of(int(band), int(column), tile_km)


def tile_offsets(tile_km: float) -> np.ndarray:
    """Return where each band starts when every tile is counted in one run.

    Args:
        tile_km: The side a tile is sized to, in kilometres.

    Returns:
        offsets: The place of each band's first tile in `every_tile`.
    """
    return np.concatenate(([0], np.cumsum(band_columns(tile_km))[:-1]))


def tile_group_name(tile: Tile, tile_km: float, tile_group_deg: float) -> str:
    """Return the group one tile is grouped into.

    Args:
        tile: The tile to place.
        tile_km: The side a tile is sized to, in kilometres.
        tile_group_deg: The side a group is sized to, in degrees.

    Returns:
        name: The group, by its row of bands and its sector of longitude.
    """
    height = HALF_TURN / len(band_columns(tile_km))
    rows = max(1, round(tile_group_deg / height))
    sectors = max(1, round(TURN / tile_group_deg))
    span = TURN if tile.circles_a_pole else tile.east_lon - tile.west_lon
    centre = (tile.west_lon + span / 2.0) % TURN
    sector = min(int(centre / TURN * sectors), sectors - 1)
    return f"r{tile.band // rows:02d}_s{sector:02d}"


def every_tile_group(tile_km: float, tile_group_deg: float) -> list[TileGroup]:
    """Return every group the tiles are grouped into.

    Args:
        tile_km: The side a tile is sized to, in kilometres.
        tile_group_deg: The side a group is sized to, in degrees.

    Returns:
        groups: Every group, south to north and west to east, each bounded by the
            tiles it holds.
    """
    grouped: dict[str, list[Tile]] = {}
    for tile in every_tile(tile_km):
        name = tile_group_name(tile, tile_km, tile_group_deg)
        grouped.setdefault(name, []).append(tile)
    groups = []
    for name, tiles in grouped.items():
        circling = any(tile.circles_a_pole for tile in tiles)
        groups.append(
            TileGroup(
                name=name,
                tiles=tuple(tiles),
                min_lat=min(tile.min_lat for tile in tiles),
                max_lat=max(tile.max_lat for tile in tiles),
                west_lon=0.0 if circling else min(tile.west_lon for tile in tiles),
                east_lon=0.0 if circling else max(tile.east_lon for tile in tiles),
            )
        )
    return groups
