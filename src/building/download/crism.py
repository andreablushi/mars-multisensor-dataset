"""Bringing one CRISM observation down from ODE into the cache."""

from __future__ import annotations

from pathlib import Path

import httpx

from building.common.pds import labels
from building.configs import crism as configs
from building.download import archive

# What ODE publishes CRISM under.
ODE = {"ihid": "MRO", "iid": "CRISM"}

# The ODE product types an observation and its geometry are published under.
TYPES = {configs.OBSERVATION: "TRDR", configs.GEOMETRY: "DDR"}

# The ODE product type a wavelength file is published under.
WAVELENGTH_TYPE = "CDR"


def fetch(observation_id: str, client: httpx.Client) -> None:
    """Bring both detectors of one observation down, or leave what is here.

    Args:
        observation_id: The observation to fetch.
        client: The client whose connections every query is asked over.

    Returns:
        None.

    Raises:
        FileNotFoundError: When ODE offers no download for a product.
        KeyError: When a label names no wavelength file.
    """
    for detector in configs.DETECTORS:
        for kind, product_type in TYPES.items():
            product_id = configs.NAMING.product(observation_id, kind, detector=detector)
            archive.collect(
                client,
                product_id,
                configs.CACHE.files(observation_id, product_id, kind),
                pt=product_type,
                **ODE,
            )
    found = {
        detector: configs.CACHE.files(
            observation_id,
            configs.NAMING.product(
                observation_id, configs.OBSERVATION, detector=detector
            ),
        )[".lbl"]
        for detector in configs.DETECTORS
    }
    # Only now do the labels exist to be asked which file calibrated them.
    for label in found.values():
        name = Path(labels.load(label)[configs.WAVELENGTH_KEY]).stem
        archive.collect(
            client,
            name,
            configs.CACHE.files(configs.WAVELENGTH_DIR, name.lower()),
            pt=WAVELENGTH_TYPE,
            **ODE,
        )
