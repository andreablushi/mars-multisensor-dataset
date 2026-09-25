"""Downloading one raw CTX scan from ODE into the cache, and placing it with ISIS."""

from __future__ import annotations

import json
import time

import httpx

from building.configs import ctx as configs
from building.download import archive
from building.preprocessing.ctx.isis import run_isis
from common.disk.files import atomic_path
from common.fetch.gate import Gate
from common.models.tile import Tile

# What ODE publishes CTX under.
ODE = {"ihid": "MRO", "iid": "CTX"}

# The ODE product type the raw scan is published under, the only one fetched.
PRODUCT_TYPE = "EDR"

# The scan's files and metadata, which carries the geometry it was taken at.
FIELDS = "fopm"

SPICE_DEADLINE = 1800.0

SPICE_REFUSED = "talking to the server"

SPICE = Gate()


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
        client, productid=identifier, pt=PRODUCT_TYPE, results=FIELDS, **ODE
    )
    if not entries:
        raise FileNotFoundError(f"ODE carries no raw scan for {identifier}.")
    acquisition = {
        key: str(entries[0][key])
        for key in configs.ODE_ACQUISITION
        if entries[0].get(key)
    }
    with atomic_path(metadata) as tmp:
        tmp.write_text(json.dumps(acquisition))
    offered = archive.file_fields(entries[0])
    archive.download_files(
        {configs.IMAGE_SUFFIX: raw},
        {configs.IMAGE_SUFFIX: offered.get(f"{identifier}{configs.IMAGE_SUFFIX}")},
        client=client,
    )


def place(identifier: str) -> None:
    """Import a fetched raw scan into ISIS and place it with the SPICE server.

    Args:
        identifier: The observation to place, its raw scan already fetched.

    Raises:
        RuntimeError: When ISIS fails to import it, or is refused past the deadline.
    """
    cube = configs.CACHE.files(identifier, identifier)[configs.CUBE_SUFFIX]
    if cube.exists():
        return
    raw = cube.with_suffix(configs.IMAGE_SUFFIX)
    staged = cube.with_suffix(f".staged{configs.CUBE_SUFFIX}")
    run_isis("mroctx2isis", {"from": raw, "to": staged})
    # While the server refuses, one lane probes it and the rest wait to be woken
    give_up_at = time.monotonic() + SPICE_DEADLINE
    while True:
        if not SPICE.wait(give_up_at):
            raise RuntimeError("spiceinit: the SPICE server refused past the deadline")
        try:
            run_isis("spiceinit", {"from": staged, "web": "yes"})
        except RuntimeError as error:
            if SPICE_REFUSED not in str(error):
                SPICE.answered()
                raise
            SPICE.refused()
        else:
            SPICE.answered()
            break
    staged.replace(cube)
    raw.unlink()
