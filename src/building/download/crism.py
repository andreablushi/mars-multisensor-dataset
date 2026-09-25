"""Downloading one CRISM observation from ODE into the cache."""

from __future__ import annotations

from pathlib import Path

import httpx

from building.configs import crism as configs
from building.download import archive
from common.models.tile import Tile
from common.pds import labels

# What ODE publishes CRISM under.
ODE = {"ihid": "MRO", "iid": "CRISM"}

# The ODE product types an observation and its geometry are published under.
PRODUCT_TYPES = {configs.Kind.OBSERVATION: "TRDR", configs.Kind.GEOMETRY: "DDR"}

# The ODE product type a wavelength file is published under.
WAVELENGTH_PRODUCT_TYPE = "CDR"


def detector_published(
    identifier: str, detector: configs.Detector, client: httpx.Client
) -> bool:
    """Download one detector's scan and geometry, telling whether ODE publishes it.

    Args:
        identifier: The observation to fetch.
        detector: The detector to download.
        client: The client every query and download goes over.

    Returns:
        published: True once both landed, False when ODE publishes no such scan.

    Raises:
        FileNotFoundError: When the scan is published but its geometry is not.
    """
    for kind, product_type in PRODUCT_TYPES.items():
        product_id = configs.NAMING.product(identifier, kind, detector=detector)
        try:
            archive.download_product(
                client,
                product_id,
                configs.CACHE.files(identifier, product_id, kind),
                pt=product_type,
                **ODE,
            )
        except FileNotFoundError:
            if kind != configs.Kind.OBSERVATION:
                raise
            return False
    return True


def fetch(identifier: str, client: httpx.Client, frames: tuple[Tile, ...]) -> None:
    """Download every published detector of one observation, and its wavelength file.

    Args:
        identifier: The observation to fetch.
        client: The client every query and download goes over.
        frames: Unused, since the observation is fetched whole.

    Raises:
        FileNotFoundError: When ODE publishes neither detector, or a geometry is
            missing.
        KeyError: When a label names no wavelength file.
    """
    # A small share of the survey was archived as one half alone
    found = [
        detector
        for detector in configs.Detector
        if detector_published(identifier, detector, client)
    ]
    if not found:
        raise FileNotFoundError(f"ODE publishes no detector of {identifier}.")
    # Only now do the labels exist to be asked which file calibrated them.
    for detector in found:
        scan = configs.NAMING.product(
            identifier, configs.Kind.OBSERVATION, detector=detector
        )
        label = configs.CACHE.files(identifier, scan)[".lbl"]
        name = Path(labels.load(label)[configs.WAVELENGTH_KEY]).stem
        archive.download_product(
            client,
            name,
            configs.CACHE.files(configs.WAVELENGTH_DIR, name.lower()),
            pt=WAVELENGTH_PRODUCT_TYPE,
            **ODE,
        )
