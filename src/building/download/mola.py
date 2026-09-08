"""Bringing down the gridded record one feature's ground is mosaicked from."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from building.configs import mola as configs
from building.download import archive

if TYPE_CHECKING:
    from shared.models.feature import Feature

# What ODE publishes MOLA under.
ODE = {"ihid": "MGS", "iid": "MOLA"}

# The ODE product type the gridded record is published under.
PRODUCT_TYPE = "MEGDR"

# ODE names a gridded product by its image file, suffix included.
ODE_SUFFIX = ".img"

# Each tile's extent beside its files, so no feature is queried for on its own.
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
        published: Where each file is served from and the ground its product covers, by
            lowercase name.
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


def grids(feature: Feature, client: httpx.Client) -> list[str]:
    """Read which grid one feature's ground is mosaicked from.

    Args:
        feature: The frame of the feature the grid has to cover.
        client: The client whose connections a query would be asked over.

    Returns:
        grids: The one grid that covers it, since a merge is never joined across two.
    """
    if feature.max_lat > configs.TILED_REACH:
        held = feature.min_lat >= configs.CAP_FLOOR
        return [configs.NORTH_CAP if held else configs.COARSE]
    if feature.min_lat < -configs.TILED_REACH:
        held = feature.max_lat <= -configs.CAP_FLOOR
        return [configs.SOUTH_CAP if held else configs.COARSE]
    return [configs.CYLINDRICAL]


def tiles(grid: str, client: httpx.Client) -> list[str]:
    """Read which tiles one grid is published as.

    Args:
        grid: The grid, as `configs.GRIDS` names it.
        client: The client whose connections the query is asked over.

    Returns:
        tiles: The tile ids the height is published for, sorted and without repeats, and
            none for a grid published whole.
    """
    held = configs.GRIDS[grid]
    if held.product:
        return []
    found = set()
    for name in record(client):
        if not name.endswith(ODE_SUFFIX):
            continue
        # Keep only wanted tiles, which drops the polar stereographic ones.
        parts = configs.NAMING.parts(Path(name).stem)
        if not parts or not parts["marker"]:
            continue
        if configs.RESOLUTIONS[parts["step"]] == held.resolution:
            found.add(parts["tile"])
    return sorted(found)


def fetch(grid: str, client: httpx.Client) -> None:
    """Bring down everything one grid is published as, or leave what is here.

    Args:
        grid: The grid to fetch, as `configs.GRIDS` names it.
        client: The client whose connections the query is asked over.

    Raises:
        FileNotFoundError: When ODE offers no download for one of them.
    """
    held = configs.GRIDS[grid]
    # A cap is a single product, so the grid's own name is the directory it lands in.
    wanted = (
        [(grid, held.product)]
        if held.product
        else [
            (tile, configs.NAMING.product(tile, configs.TOPOGRAPHY))
            for tile in tiles(grid, client)
        ]
    )
    for directory, product in wanted:
        files = configs.CACHE.files(directory, product, configs.TOPOGRAPHY)
        if all(path.exists() for path in files.values()):
            continue
        # One product carries many features, so only the first to want it fetches.
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
