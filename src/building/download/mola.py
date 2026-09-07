"""Bringing down every MOLA gridded tile one feature's ground falls on."""

from __future__ import annotations

from collections import defaultdict
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

# How fine a grid to read. MEGDR publishes 4 to 128, and only 128 beats a kilometre.
RESOLUTION = 128

Box = tuple[float, float, float, float]

# The whole record, under a hundred and unchanging, so it is read once for a run.
_RECORD: dict[str, tuple[str, Box]] = {}


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


def tiles(feature: FeatureFrame, client: httpx.Client) -> list[str]:
    """Read which tiles hold one feature's ground.

    The gridded record carries no per observation time, so the selection never
    names a tile and each tile's own extent is what the feature is matched to.

    Args:
        feature: The frame of the feature whose ground the tiles have to cover.
        client: The client whose connections the query is asked over.

    Returns:
        The tile ids both planes are published for, sorted and without repeats.
    """
    # A feature at a pole reaches every longitude, one over the meridian is two runs.
    if feature.west_lon == feature.east_lon:
        spans = ((0.0, 360.0),)
    elif feature.west_lon > feature.east_lon:
        spans = ((feature.west_lon, 360.0), (0.0, feature.east_lon))
    else:
        spans = ((feature.west_lon, feature.east_lon),)
    found: dict[str, set[str]] = defaultdict(set)
    for name, (_, covers) in record(client).items():
        if not name.endswith(ODE_SUFFIX):
            continue
        # Keep only wanted tiles, which drops the polar stereographic ones.
        tile = configs.NAMING.parse(Path(name).stem)
        if not tile or configs.resolution(tile) != RESOLUTION:
            continue
        min_lat, max_lat, west_lon, east_lon = covers
        if feature.min_lat > max_lat or min_lat > feature.max_lat:
            continue
        if any(west <= east_lon and west_lon <= east for west, east in spans):
            found[tile].add(Path(name).stem)
    return sorted(
        tile
        for tile, seen in found.items()
        if seen.issuperset(configs.NAMING.product(tile, kind) for kind in configs.KINDS)
    )


def fetch(tile: str, client: httpx.Client) -> None:
    """Bring both planes of one tile down, or leave what is here.

    Args:
        tile: The tile to fetch, such as 00n180hb.
        client: The client whose connections the query is asked over.

    Returns:
        None.

    Raises:
        FileNotFoundError: When ODE offers no download for a plane.
    """
    wanted = {
        kind: configs.CACHE.files(tile, configs.NAMING.product(tile, kind), kind)
        for kind in configs.KINDS
    }
    if any(not path.exists() for files in wanted.values() for path in files.values()):
        # A gridded product has no id, so it is reached by the file it is published as.
        offered = record(client)
        for kind, files in wanted.items():
            product = configs.NAMING.product(tile, kind)
            archive.bring(
                files,
                {
                    Path(name).suffix: url
                    for name, (url, _) in offered.items()
                    if Path(name).stem == product
                },
                client=client,
            )
