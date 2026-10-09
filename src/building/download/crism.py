"""Downloading one CRISM observation from ODE into the cache."""

from __future__ import annotations

import httpx

from building.configs import crism as configs
from building.download import archive
from common.models.tile import Tile


def download_detector_product(
    identifier: str,
    detector: configs.Detector,
    kind: configs.Kind,
    client: httpx.Client,
) -> None:
    """Download one product one detector of an observation was published as.

    Args:
        identifier: The observation to fetch.
        detector: The detector the product belongs to.
        kind: Which of its products to download, its scan or its geometry.
        client: The client every query and download goes over.

    Raises:
        FileNotFoundError: When ODE publishes no such product.
    """
    product_id = configs.NAMING.product(identifier, kind, detector=detector)
    archive.download_product(
        client,
        product_id,
        configs.CACHE.files(identifier, product_id, kind),
        pt=configs.PRODUCT_TYPES[kind],
        **configs.ODE,
    )


def fetch(identifier: str, client: httpx.Client, frames: tuple[Tile, ...]) -> None:
    """Download every published detector of one observation and its geometry.

    Args:
        identifier: The observation to fetch.
        client: The client every query and download goes over.
        frames: Unused, since the observation is fetched whole.

    Raises:
        FileNotFoundError: When ODE publishes neither detector, or the geometry of
            the first is missing.
    """
    # A small share of the survey was archived as one half alone
    found = []
    for detector in configs.Detector:
        try:
            download_detector_product(
                identifier, detector, configs.Kind.OBSERVATION, client
            )
        except FileNotFoundError:
            continue
        found.append(detector)
    if not found:
        raise FileNotFoundError(f"ODE publishes no detector of {identifier}.")
    # Every half is placed by the first one's geometry, so no other is fetched
    download_detector_product(identifier, found[0], configs.Kind.GEOMETRY, client)
