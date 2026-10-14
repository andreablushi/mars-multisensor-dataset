"""Downloading one MOLA grid from ODE into the cache, and picking a tile's grid."""

from __future__ import annotations

import threading
from pathlib import Path

import httpx

from building.configs import mola as configs
from building.download import archive
from building.preprocessing.common import cut
from building.preprocessing.mola import projection
from common.models.tile import Tile
from common.pds import labels

# ODE names a gridded product by its image file, suffix included.
ODE_SUFFIX = ".img"

LABEL_SUFFIX = ".lbl"

# How many to ask at once. The record is under a hundred, so one page holds it all.
PAGE = 500

# The whole record, under a hundred and unchanging, so it is read once for a run.
_RECORD: dict[str, str] = {}

_PRODUCT_LOCKS: dict[str, threading.Lock] = {}
_PRODUCT_LOCKS_GUARD = threading.Lock()


def record_files(client: httpx.Client) -> dict[str, str]:
    """Read every file of the gridded record, asking ODE only on the first call.

    Args:
        client: The client the query goes over.

    Returns:
        files: Each file's URL, keyed by lowercase name.
    """
    if not _RECORD:
        for entry in archive.query_products(
            client,
            pt=configs.PRODUCT_TYPE,
            limit=str(PAGE),
            results=configs.ODE_RESULTS,
            **configs.ODE,
        ):
            _RECORD.update(archive.file_fields(entry))
    return _RECORD


def grid_sheets(resolution: int, client: httpx.Client) -> list[str]:
    """Read which sheets a sheeted grid of one resolution is published as.

    Args:
        resolution: How many bins of the grid one degree holds.
        client: The client the query goes over.

    Returns:
        sheets: The sorted unique sheet ids.
    """
    # Keep only wanted sheets, which drops the polar stereographic ones.
    sheets = {
        configs.NAMING.observation_id(Path(name).stem)
        for name in record_files(client)
        if name.endswith(ODE_SUFFIX)
    }
    return sorted(
        sheet
        for sheet in sheets
        if sheet and configs.sheet_resolution(sheet) == resolution
    )


def fetch(identifier: str, client: httpx.Client, frames: tuple[Tile, ...]) -> None:
    """Download every product of one grid that reaches a tile, skipping those on disk.

    Args:
        identifier: The grid to fetch, as `configs.GRIDS` names it.
        client: The client every query and download goes over.
        frames: The tiles it is cut to, the only ground whose images are downloaded.

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
            urls = {
                Path(name).suffix: url
                for name, url in record_files(client).items()
                if Path(name).stem == product
            }
            label = files[LABEL_SUFFIX]
            archive.download_files({LABEL_SUFFIX: label}, urls, client=client)
            placed = projection.grid_position(labels.load(label))
            if any(cut.overlap(placed, frame) is not None for frame in frames):
                archive.download_files(files, urls, client=client)


def mola_grid(tile: Tile) -> str:
    """Read which grid one tile's ground is mosaicked from.

    Args:
        tile: The frame of the tile the grid has to cover.

    Returns:
        grid: The one grid that covers it, since a merge is never joined across two.
    """
    if tile.max_lat > configs.SHEETED_REACH:
        return configs.NORTH_POLAR
    if tile.min_lat < -configs.SHEETED_REACH:
        return configs.SOUTH_POLAR
    return configs.EQUATORIAL
