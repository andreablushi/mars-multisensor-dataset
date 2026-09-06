"""Bringing one SHARAD radargram down from ODE into the cache."""

from __future__ import annotations

import httpx

from building.configs import sharad as configs
from building.download import archive

# What ODE publishes SHARAD under.
ODE = {"ihid": "MRO", "iid": "SHARAD"}

# The ODE product types a radargram and its geometry are published under.
TYPES = {configs.OBSERVATION: "USRDRV2", configs.GEOMETRY: "USGEOMV2"}


def fetch(observation_id: str, client: httpx.Client) -> None:
    """Bring one radargram and its geometry down, or leave what is here.

    Args:
        observation_id: The observation to fetch.
        client: The client whose connections every query is asked over.

    Returns:
        None.

    Raises:
        FileNotFoundError: When ODE offers no download for a product.
    """
    for kind, product_type in TYPES.items():
        product_id = configs.NAMING.product(observation_id, kind)
        archive.collect(
            client,
            product_id,
            configs.CACHE.files(observation_id, product_id, kind),
            pt=product_type,
            **ODE,
        )
