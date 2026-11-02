"""Downloading one raw CTX scan from ODE into the cache, and placing it with ISIS."""

from __future__ import annotations

import csv
from datetime import datetime

import httpx

from building.configs import ctx as configs
from building.download import archive, kernels
from building.preprocessing.ctx.isis import read_cube_label, run_isis
from common.disk.files import write_json
from common.models.tile import Tile


def fetch(identifier: str, client: httpx.Client, frames: tuple[Tile, ...]) -> None:
    """Download one raw scan and what ODE says of it, leaving it to be placed.

    Args:
        identifier: The observation to fetch.
        client: The client every query and download goes over.
        frames: Unused, since the observation is fetched whole.

    Raises:
        FileNotFoundError: When ODE carries no raw scan of that name.
    """
    files = configs.CACHE.files(identifier, identifier)
    cube, metadata = files[configs.CUBE_SUFFIX], files[configs.METADATA_SUFFIX]
    raw = cube.with_suffix(configs.IMAGE_SUFFIX)
    if (cube.exists() or raw.exists()) and metadata.exists():
        return
    entries = archive.query_products(
        client,
        productid=identifier,
        pt=configs.PRODUCT_TYPE,
        results=configs.ODE_RESULTS,
        **configs.ODE,
    )
    if not entries:
        raise FileNotFoundError(f"ODE carries no raw scan for {identifier}.")
    acquisition = {
        key: str(entries[0][key])
        for key in configs.ODE_ACQUISITION
        if entries[0].get(key)
    }
    write_json(metadata, acquisition | index_azimuths(entries[0]))
    offered = archive.file_fields(entries[0])
    archive.download_files(
        {configs.IMAGE_SUFFIX: raw},
        {configs.IMAGE_SUFFIX: offered.get(f"{identifier}{configs.IMAGE_SUFFIX}")},
        client=client,
    )


def index_azimuths(entry: dict) -> dict[str, str]:
    """Return the image plane azimuths of the PDS index row ODE quotes for a scan.

    Args:
        entry: The scan's product, as ODE answers with it.

    Returns:
        azimuths: Each of `configs.INDEX_AZIMUTHS` as its row writes it, none
            without that row.
    """
    notes = entry.get("ODE_notes", {}).get("ODE_note", [])
    if configs.INDEX_NOTE not in notes:
        return {}
    row = next(csv.reader([notes[notes.index(configs.INDEX_NOTE) + 1]]))
    return {key: row[at].strip() for key, at in configs.INDEX_AZIMUTHS.items()}


def place(identifier: str, client: httpx.Client) -> None:
    """Import a fetched raw scan into ISIS and attach the SPICE kernels it was taken at.

    Args:
        identifier: The observation to place, its raw scan already fetched.
        client: The client every kernel is downloaded over.

    Raises:
        RuntimeError: When ISIS fails to import or place it.
    """
    cube = configs.CACHE.files(identifier, identifier)[configs.CUBE_SUFFIX]
    if cube.exists():
        return
    raw = cube.with_suffix(configs.IMAGE_SUFFIX)
    staged = cube.with_suffix(f".staged{configs.CUBE_SUFFIX}")
    run_isis("mroctx2isis", {"from": raw, "to": staged})
    start = datetime.fromisoformat(read_cube_label(staged)["StartTime"])
    with kernels.held_kernels(start.replace(tzinfo=None), client):
        run_isis("spiceinit", {"from": staged})
    staged.replace(cube)
    raw.unlink()
