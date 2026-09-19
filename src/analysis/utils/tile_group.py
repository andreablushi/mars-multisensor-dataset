"""Gathering the tiles of the grid into the groups ODE is asked about at once."""

from __future__ import annotations

from analysis.models.tile_group import TileGroup
from common.maths.geodesy import HALF_TURN, TURN, longitude_span
from common.maths.tessellate import Tessellate
from common.models.tile import Tile


def group_name(bands: int, tile: Tile, tile_group_deg: float) -> str:
    """Return the group one tile is grouped into.

    Args:
        bands: How many bands the grid the tile sits on is split into.
        tile: The tile to place.
        tile_group_deg: The side a group is sized to, in degrees.

    Returns:
        name: The group, by its row of bands and its sector of longitude.
    """
    rows = max(1, round(tile_group_deg / (HALF_TURN / bands)))
    sectors = max(1, round(TURN / tile_group_deg))
    span = longitude_span(tile.west_lon, tile.east_lon)
    centre = (tile.west_lon + span / 2.0) % TURN
    sector = min(int(centre / TURN * sectors), sectors - 1)
    return f"r{tile.band // rows:02d}_s{sector:02d}"


def every_tile_group(grid: Tessellate, tile_group_deg: float) -> list[TileGroup]:
    """Return every group the tiles of one grid are grouped into.

    Args:
        grid: The grid whose tiles are grouped.
        tile_group_deg: The side a group is sized to, in degrees.

    Returns:
        groups: Every group, south to north and west to east, each bounded by the
            tiles it holds.
    """
    bands = len(grid.columns)
    grouped: dict[str, list[Tile]] = {}
    for band, count in enumerate(grid.columns):
        for column in range(count):
            tile = grid.tile_of(band, column)
            grouped.setdefault(group_name(bands, tile, tile_group_deg), []).append(tile)
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
