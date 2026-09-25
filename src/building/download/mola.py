"""Bringing down the gridded record one tile's ground is mosaicked from."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from building.configs import mola as configs
from building.download import archive

if TYPE_CHECKING:
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

_FETCHING: dict[str, threading.Lock] = {}
_GUARD = threading.Lock()


def record(client: httpx.Client) -> dict[str, tuple[str, Box]]:
    """Read the whole gridded record, once per run.

    Args:
        client: The client whose connections the query is asked over.

    Returns:
        published: Each file's URL and the ground it covers, by lowercase name.
    """
    if not _RECORD:
        for entry in archive.query(
            client, pt=PRODUCT_TYPE, limit=str(PAGE), results=FIELDS, **ODE
        ):
            covers = (
                float(entry["Minimum_latitude"]),
                float(entry["Maximum_latitude"]),
                float(entry["Westernmost_longitude"]),
                float(entry["Easternmost_longitude"]),
            )
            for name, url in archive.published(entry).items():
                _RECORD[name] = (url, covers)
    return _RECORD


def grids(tile: Tile, client: httpx.Client) -> list[str]:
    """Read which grid one tile's ground is mosaicked from.

    Args:
        tile: The frame of the tile the grid has to cover.
        client: The client whose connections a query would be asked over.

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


def sheets(grid: str, client: httpx.Client) -> list[str]:
    """Read which sheets one grid is published as.

    Args:
        grid: The grid, as `configs.GRIDS` names it.
        client: The client whose connections the query is asked over.

    Returns:
        sheets: The sorted unique sheet ids, and none for a whole grid.
    """
    held = configs.GRIDS[grid]
    if held.product:
        return []
    found = set()
    for name in record(client):
        if not name.endswith(ODE_SUFFIX):
            continue
        # Keep only wanted sheets, which drops the polar stereographic ones.
        parts = configs.NAMING.parts(Path(name).stem)
        if not parts or not parts["marker"]:
            continue
        if configs.RESOLUTIONS[parts["step"]] == held.resolution:
            found.add(parts["sheet"])
    return sorted(found)


def fetch(grid: str, client: httpx.Client, frames: tuple[Tile, ...]) -> None:
    """Bring down everything one grid is published as, or leave what is here.

    Args:
        grid: The grid to fetch, as `configs.GRIDS` names it.
        client: The client whose connections the query is asked over.
        frames: The tiles it is cut to, which take it whole.

    Raises:
        FileNotFoundError: When ODE offers no download for one of them.
    """
    held = configs.GRIDS[grid]
    # A cap is a single product, so the grid's own name is the directory it lands in.
    wanted = (
        [(grid, held.product)]
        if held.product
        else [
            (sheet, configs.NAMING.product(sheet, configs.TOPOGRAPHY))
            for sheet in sheets(grid, client)
        ]
    )
    for directory, product in wanted:
        files = configs.CACHE.files(directory, product, configs.TOPOGRAPHY)
        if all(path.exists() for path in files.values()):
            continue
        # One product carries many tiles, so only the first to want it fetches.
        with _GUARD:
            fetching = _FETCHING.setdefault(product, threading.Lock())
        with fetching:
            if all(path.exists() for path in files.values()):
                continue
            offered = record(client)
            archive.bring(
                files,
                {
                    Path(name).suffix: url
                    for name, (url, _) in offered.items()
                    if Path(name).stem == product
                },
                client=client,
            )
