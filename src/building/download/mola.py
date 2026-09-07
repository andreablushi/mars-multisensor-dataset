"""Bringing down the gridded record one feature's ground is mosaicked from."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from building.configs import mola as configs
from building.download import archive

if TYPE_CHECKING:
    from building.models.feature import FeatureFrame

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
        Where each published file is served from and the ground its product
        covers, keyed by the file's own lowercase name.
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


def grids(feature: FeatureFrame, client: httpx.Client) -> list[str]:
    """Read which grid one feature's ground is mosaicked from.

    Args:
        feature: The frame of the feature the grid has to cover.
        client: The client whose connections a query would be asked over.

    Returns:
        The one grid that covers it, since a mosaic is never joined across two.
    """
    return [configs.CYLINDRICAL]


def tiles(grid: str, client: httpx.Client) -> list[str]:
    """Read which tiles one grid is published as.

    Args:
        grid: The grid, as `configs.GRIDS` names it.
        client: The client whose connections the query is asked over.

    Returns:
        The tile ids the height is published for, sorted and without repeats.
    """
    resolution = configs.GRIDS[grid]
    found = set()
    for name in record(client):
        if not name.endswith(ODE_SUFFIX):
            continue
        # Keep only wanted tiles, which drops the polar stereographic ones.
        tile = configs.NAMING.parse(Path(name).stem)
        if tile and configs.resolution(tile) == resolution:
            found.add(tile)
    return sorted(found)


def fetch(grid: str, client: httpx.Client) -> None:
    """Bring down every tile of one grid, or leave what is here.

    Args:
        grid: The grid to fetch, as `configs.GRIDS` names it.
        client: The client whose connections the query is asked over.

    Returns:
        None.

    Raises:
        FileNotFoundError: When ODE offers no download for a tile.
    """
    for tile in tiles(grid, client):
        product = configs.NAMING.product(tile, configs.TOPOGRAPHY)
        wanted = configs.CACHE.files(tile, product, configs.TOPOGRAPHY)
        if all(path.exists() for path in wanted.values()):
            continue
        # One tile carries many features, so only the first to want it fetches.
        with _GUARD:
            held = _FETCHING.setdefault(tile, threading.Lock())
        with held:
            if all(path.exists() for path in wanted.values()):
                continue
            offered = record(client)
            archive.bring(
                wanted,
                {
                    Path(name).suffix: url
                    for name, (url, _) in offered.items()
                    if Path(name).stem == product
                },
                client=client,
            )
