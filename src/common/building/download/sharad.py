"""Bringing one SHARAD radargram down from ODE into the cache."""

from __future__ import annotations

import httpx
import numpy as np

from common.building.common.pds import labels
from common.building.configs import sharad as configs
from common.building.download import archive

# What ODE publishes SHARAD under.
ODE = {"ihid": "MRO", "iid": "SHARAD"}

# The ODE product types a radargram, its geometry and its clutter are published under.
TYPES = {
    configs.OBSERVATION: "USRDRV2",
    configs.GEOMETRY: "USGEOMV2",
    configs.CLUTTER: "SHSIMU",
}


def fetch(observation_id: str, client: httpx.Client) -> None:
    """Bring one radargram, its geometry and its clutter down, or leave what is here.

    Args:
        observation_id: The observation to fetch.
        client: The client whose connections every query is asked over.

    Raises:
        FileNotFoundError: When ODE offers no download for a product.
        FetchError: When the archive will not serve the combined clutter alone.
    """
    for kind, product_type in TYPES.items():
        product_id = configs.NAMING.product(observation_id, kind)
        span = None
        if kind == configs.CLUTTER:
            radargram = configs.CACHE.files(
                observation_id,
                configs.NAMING.product(observation_id, configs.OBSERVATION),
                configs.OBSERVATION,
            )
            lines, samples, *_ = labels.layout(labels.load(radargram[".lbl"]))
            size = lines * samples * np.dtype(configs.CLUTTER_TYPE).itemsize
            span = (configs.CLUTTER_ARRAY * size, (configs.CLUTTER_ARRAY + 1) * size)
        archive.collect(
            client,
            product_id,
            configs.CACHE.files(observation_id, product_id, kind),
            span=span,
            pt=product_type,
            **ODE,
        )
