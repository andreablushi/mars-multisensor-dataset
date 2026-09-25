"""Bringing one raw CTX scan down, and placing it with ISIS once it has landed."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import httpx

from building.configs import ctx as configs
from building.download import archive
from building.preprocessing.ctx.isis import run_isis
from common.disk.files import atomic_path
from common.fetch import http

if TYPE_CHECKING:
    from common.models.tile import Tile

# What ODE publishes CTX under, the raw scan being the only type it carries.
ODE = {"ihid": "MRO", "iid": "CTX", "pt": "EDR"}

# The scan's files and metadata, which carries the geometry it was taken at.
FIELDS = "fopm"

SPICE_RETRIES = 5

SPICE_BACKOFF = 5.0

SPICE_REFUSED = "talking to the server"


def fetch(observation_id: str, client: httpx.Client, frames: tuple[Tile, ...]) -> None:
    """Bring the raw scan and what ODE says of it, leaving it to be placed.

    Args:
        observation_id: The observation to fetch.
        client: The client whose connections every query is asked over.
        frames: The tiles it is cut to, which take it whole.

    Raises:
        FileNotFoundError: When ODE carries no raw scan of that name.
    """
    files = configs.CACHE.files(observation_id, observation_id)
    cube, said = files[configs.CUBE_SUFFIX], files[configs.METADATA_SUFFIX]
    raw = cube.with_suffix(configs.IMAGE_SUFFIX)
    if (cube.exists() or raw.exists()) and said.exists():
        return
    entries = archive.query(client, productid=observation_id, results=FIELDS, **ODE)
    if not entries:
        raise FileNotFoundError(f"ODE carries no raw scan for {observation_id}.")
    acquisition = {
        key: str(entries[0][key])
        for key in configs.ODE_ACQUISITION
        if entries[0].get(key)
    }
    with atomic_path(said) as tmp:
        tmp.write_text(json.dumps(acquisition))
    offered = archive.published(entries[0])
    archive.bring(
        {configs.IMAGE_SUFFIX: raw},
        {configs.IMAGE_SUFFIX: offered.get(f"{observation_id}{configs.IMAGE_SUFFIX}")},
        client=client,
    )


def place(observation_id: str) -> None:
    """Import a fetched raw scan into ISIS and place it with the SPICE server.

    Args:
        observation_id: The observation to place, its raw scan already fetched.

    Raises:
        RuntimeError: When ISIS fails to import it, or to place it after every retry.
    """
    cube = configs.CACHE.files(observation_id, observation_id)[configs.CUBE_SUFFIX]
    if cube.exists():
        return
    raw = cube.with_suffix(configs.IMAGE_SUFFIX)
    staged = cube.with_suffix(f".staged{configs.CUBE_SUFFIX}")
    run_isis("mroctx2isis", {"from": raw, "to": staged})
    for attempt in range(SPICE_RETRIES + 1):
        if attempt:
            http.slept(attempt, SPICE_BACKOFF)
        try:
            run_isis("spiceinit", {"from": staged, "web": "yes"})
            break
        except RuntimeError as error:
            if SPICE_REFUSED not in str(error) or attempt == SPICE_RETRIES:
                raise
    staged.replace(cube)
    raw.unlink()
