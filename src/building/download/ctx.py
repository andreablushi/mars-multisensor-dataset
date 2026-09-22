"""Bringing one raw CTX scan down, and placing it with ISIS as it lands."""

from __future__ import annotations

import json

import httpx

from building.common.isis import run_isis
from building.configs import ctx as configs
from building.download import archive
from common.disk.files import atomic_path
from common.fetch import http

# What ODE publishes CTX under, the raw scan being the only type it carries.
ODE = {"ihid": "MRO", "iid": "CTX", "pt": "EDR"}

# The scan's files and metadata, which carries the geometry it was taken at.
FIELDS = "fopm"

SPICE_RETRIES = 5

SPICE_BACKOFF = 5.0

SPICE_REFUSED = "talking to the server"


def fetch(observation_id: str, client: httpx.Client) -> None:
    """Bring the raw scan and what ODE says of it, then import and place it.

    Args:
        observation_id: The observation to fetch.
        client: The client whose connections every query is asked over.

    Raises:
        FileNotFoundError: When ODE carries no raw scan of that name.
        RuntimeError: When ISIS fails to import it, or to place it after every retry.
    """
    files = configs.CACHE.files(observation_id, observation_id)
    cube, said = files[configs.CUBE_SUFFIX], files[configs.METADATA_SUFFIX]
    if cube.exists() and said.exists():
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
    raw = cube.with_suffix(configs.IMAGE_SUFFIX)
    offered = archive.published(entries[0])
    archive.bring(
        {configs.IMAGE_SUFFIX: raw},
        {configs.IMAGE_SUFFIX: offered.get(f"{observation_id}{configs.IMAGE_SUFFIX}")},
        client=client,
    )
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
