"""The credentials a platform run holds, which lapse long before the run ends."""

from __future__ import annotations

import os

from digitalhub.stores.client.base.factory import get_client
from dotenv import load_dotenv

from shared import paths

TOKEN = "DHCORE_PERSONAL_ACCESS_TOKEN"

MINTED_FROM = ("DHCORE_ISSUER", "DHCORE_CLIENT_ID")


def minting_envs() -> list[dict[str, str]]:
    """Return what a job is told so it can mint credentials of its own.

    Returns:
        told: The authority to ask and the client to ask as, each as the platform
            spells a variable, read from the environment or from the `.env` beside
            this file's repository. The token names neither, so a job handed it
            alone has nowhere to present it.

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


def refreshed() -> bool:
    """Mint the run's credentials again, the platform's and the store's alike.

    Returns:
        minted: Whether fresh ones were had, which a run holding a token to mint
            from always is, and a run holding only what it was started with never.
    """
    return get_client().eval_retry()
