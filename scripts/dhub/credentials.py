"""What a job is given to mint its own credentials, which lapse before a run ends."""

from __future__ import annotations

import os

from dotenv import load_dotenv

from common import paths

TOKEN = "DHCORE_PERSONAL_ACCESS_TOKEN"

MINTED_FROM = ("DHCORE_ISSUER", "DHCORE_CLIENT_ID")


def minting_envs() -> list[dict[str, str]]:
    """Return what a job is told so it can mint credentials of its own.

    Returns:
        told: The authority and client variables, read from the environment or `.env`.

    Raises:
        RuntimeError: When either is unset, which a job cannot mint without.
    """
    load_dotenv(paths.REPO_ROOT / ".env")
    told = []
    for name in MINTED_FROM:
        value = os.environ.get(name)
        if not value:
            raise RuntimeError(f"{name} is unset; see .env.example")
        told.append({"name": name, "value": value})
    return told
