"""Downloading one MOLA grid from ODE into the cache, and picking a tile's grid."""

from __future__ import annotations

import threading
from pathlib import Path

import httpx

from building.configs import mola as configs
from building.download import archive
from common.models.tile import Tile

# What ODE publishes MOLA under.
ODE = {"ihid": "MGS", "iid": "MOLA"}

# The ODE product type the gridded record is published under.
PRODUCT_TYPE = "MEGDR"

# ODE names a gridded product by its image file, suffix included.
ODE_SUFFIX = ".img"

# Each sheet's extent beside its files, so no tile is queried for on its own.
FIELDS = "opmf"

# How many to ask at once. The record is under a hundred, so one page holds it all.
PAGE = 500

Box = tuple[float, float, float, float]

# The whole record, under a hundred and unchanging, so it is read once for a run.
_RECORD: dict[str, tuple[str, Box]] = {}

_PRODUCT_LOCKS: dict[str, threading.Lock] = {}
_PRODUCT_LOCKS_GUARD = threading.Lock()


def record_files(client: httpx.Client) -> dict[str, tuple[str, Box]]:
    """Read every file of the gridded record, asking ODE only on the first call.

    Args:
        client: The client the query goes over.

    Returns:
        files: Each file's URL and the ground it covers, keyed by lowercase name.
    """
    if not _RECORD:
        for entry in archive.query_products(
            client, pt=PRODUCT_TYPE, limit=str(PAGE), results=FIELDS, **ODE
        ):
            covers = (
                float(entry["Minimum_latitude"]),
                float(entry["Maximum_latitude"]),
                float(entry["Westernmost_longitude"]),
                float(entry["Easternmost_longitude"]),
            )
            for name, url in archive.file_fields(entry).items():
                _RECORD[name] = (url, covers)
    return _RECORD


def grid_sheets(resolution: int, client: httpx.Client) -> list[str]:
    """Read which sheets a sheeted grid of one resolution is published as.

    Args:
        resolution: How many bins of the grid one degree holds.
        client: The client the query goes over.

    Returns:
        sheets: The sorted unique sheet ids.
    """
    sheets = set()
    for name in record_files(client):
        if not name.endswith(ODE_SUFFIX):
            continue
        # Keep only wanted sheets, which drops the polar stereographic ones.
        parts = configs.NAMING.parts(Path(name).stem)
        if not parts or not parts["marker"]:
            continue
        if configs.RESOLUTIONS[parts["step"]] == resolution:
            sheets.add(parts["sheet"])
    return sorted(sheets)


def fetch(identifier: str, client: httpx.Client, frames: tuple[Tile, ...]) -> None:
    """Download every product one grid is published as, skipping those on disk.

    Args:
        identifier: The grid to fetch, as `configs.GRIDS` names it.
        client: The client every query and download goes over.
        frames: Unused, since the grid is fetched whole.

    Raises:
        FileNotFoundError: When ODE offers no download for one of them.
    """
    held = configs.GRIDS[identifier]
    # A cap is a single product, so the grid's own name is the directory it lands in.
    if held.product:
        wanted = [(identifier, held.product)]
    else:
        wanted = [
            (sheet, configs.NAMING.product(sheet, configs.Kind.TOPOGRAPHY))
            for sheet in grid_sheets(held.resolution, client)
        ]
    for directory, product in wanted:
        files = configs.CACHE.files(directory, product, configs.Kind.TOPOGRAPHY)
        # One product carries many tiles, so only the first to want it fetches.
        with _PRODUCT_LOCKS_GUARD:
            lock = _PRODUCT_LOCKS.setdefault(product, threading.Lock())
        with lock:
            if all(path.exists() for path in files.values()):
                continue
            archive.download_files(
                files,
                {
                    Path(name).suffix: url
                    for name, (url, _) in record_files(client).items()
                    if Path(name).stem == product
                },
                client=client,
            )


def tile_grids(tile: Tile, client: httpx.Client) -> list[str]:
    """Read which grid one tile's ground is mosaicked from.

    Args:
        tile: The frame of the tile the grid has to cover.
        client: Unused, required by the dispatcher's `identifiers` signature.

    Returns:
        grids: The one grid that covers it, since a merge is never joined across two.
    """
    if tile.max_lat > configs.SHEETED_REACH:
        held = tile.min_lat >= configs.POLAR_FLOOR
        return [configs.NORTH_POLAR if held else configs.COARSE]
    if tile.min_lat < -configs.SHEETED_REACH:
        held = tile.max_lat <= -configs.POLAR_FLOOR
        return [configs.SOUTH_POLAR if held else configs.COARSE]
    return [configs.EQUATORIAL]
