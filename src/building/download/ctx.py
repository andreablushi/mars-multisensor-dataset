"""Bringing one projected CTX scan down from ASU into the cache."""

from __future__ import annotations

from urllib.parse import quote

import httpx

from building.configs import ctx as configs
from building.download import archive
from utils.fetch.http import FetchError

# What ODE publishes CTX under.
ODE = {"ihid": "MRO", "iid": "CTX"}

# The only type ODE carries, the raw scan. ASU builds the projected one and is asked.
PRODUCT_TYPE = "EDR"

# The scan's metadata, where the volume is published rather than buried in a URL.
FIELDS = "opm"
VOLUME_KEY = "PDSVolume_Id"

# Where ASU serves what it built, given the path the file sits at.
ASU_URL = "https://image.mars.asu.edu/stream/{name}?image={path}"

# Where ASU keeps one product of a scan, under the volume it was archived on.
ASU_PATH = "/mars/images/ctx/{volume}/{place}/{name}"

# Which ASU directory each kind is kept in.
DIRECTORIES = {configs.IMAGE: "prj_full", configs.LABEL: "stage"}

# What ASU suffixes each with: one shared image, and a label per projection it writes.
REMOTE_IMAGE = ".tiff"
REMOTE_LABELS = (".scyl.isis.hdr", ".ps.isis.hdr")

# How long to wait for the scan, which ASU builds on the way out.
TIMEOUT = 900.0


def fetch(observation_id: str, client: httpx.Client) -> None:
    """Bring the projected scan and its label down, or leave what is here.

    ASU keeps what it built under the volume the raw scan came from, and only
    ODE knows which that is. It writes the scan in whichever projection holds
    it, so the label is asked for under each in turn until one answers, and the
    pixels come down once beside it. The last projection is asked for outside
    that guard, so a scan ASU serves in none of them fails as what it is rather
    than as the projection having been guessed wrong.

    Args:
        observation_id: The observation to fetch.
        client: The client whose connections every query is asked over.

    Returns:
        None.

    Raises:
        FileNotFoundError: When ODE carries no raw scan to read the volume off.
        FetchError: When ASU serves the scan in none of the projections.
    """
    destination = configs.CACHE.files(observation_id, observation_id)
    if all(path.exists() for path in destination.values()):
        return
    entries = archive.query(
        client, productid=observation_id, results=FIELDS, pt=PRODUCT_TYPE, **ODE
    )
    archived = entries[0].get(VOLUME_KEY) if entries else None
    if not archived:
        raise FileNotFoundError(f"ODE carries no raw scan for {observation_id}.")
    volume = str(archived).lower()
    for remote in REMOTE_LABELS[:-1]:
        try:
            archive.bring(destination, _asu(observation_id, volume, remote), TIMEOUT)
            return
        except FetchError:
            continue
    archive.bring(destination, _asu(observation_id, volume, REMOTE_LABELS[-1]), TIMEOUT)


def _asu(observation_id: str, volume_id: str, label: str) -> dict[str, str]:
    """Return where ASU serves each product of one scan from.

    Args:
        observation_id: The scan to build the URLs for.
        volume_id: The PDS volume the raw scan was archived on, which is the
            directory ASU keeps what it built from it under.
        label: What ASU suffixes the label with, which says the projection the
            scan was written in.

    Returns:
        The URL each product is streamed from, keyed by its suffix on disk.
    """
    # ODE spells a scan in lower case, and ASU serves it in upper.
    scan = observation_id.upper()
    remote = {configs.IMAGE: REMOTE_IMAGE, configs.LABEL: label}
    return {
        configs.SUFFIXES[kind]: ASU_URL.format(
            name=f"{scan}{configs.SUFFIXES[kind]}",
            path=quote(
                ASU_PATH.format(
                    volume=volume_id,
                    place=DIRECTORIES[kind],
                    name=f"{scan}{remote[kind]}",
                )
            ),
        )
        for kind in configs.KINDS
    }
