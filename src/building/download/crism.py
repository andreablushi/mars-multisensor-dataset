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
    """Bring down whichever detectors of one observation ODE holds, or leave them.

    Both detectors are read out together, but a small share of the survey was
    archived as one half alone, so each is asked for and the observation is
    built from whichever answered rather than being dropped for the other.

    Args:
        observation_id: The observation to fetch.
        client: The client whose connections every query is asked over.

    Returns:
        None.

    Raises:
        FileNotFoundError: When ODE publishes neither detector, or publishes one
            without the geometry that places it.
        KeyError: When a label names no wavelength file.
    """

    def brought(detector: str) -> bool:
        """Bring one detector of the observation down, where it was archived.

        Args:
            detector: Which detector to ask ODE for.

        Returns:
            True when it landed, and False when ODE publishes no observation
            under that detector.

        Raises:
            FileNotFoundError: When the observation is published but the
                geometry placing it is not.
        """
        for kind, product_type in TYPES.items():
            product_id = configs.NAMING.product(observation_id, kind, detector=detector)
            try:
                archive.collect(
                    client,
                    product_id,
                    configs.CACHE.files(observation_id, product_id, kind),
                    pt=product_type,
                    **ODE,
                )
            except FileNotFoundError:
                if kind != configs.OBSERVATION:
                    raise
                return False
        return True

    found = [name for name in configs.DETECTORS if brought(name)]
    if not found:
        raise FileNotFoundError(f"ODE publishes no detector of {observation_id}.")
    # Only now do the labels exist to be asked which file calibrated them.
    for detector in found:
        scan = configs.NAMING.product(
            observation_id, configs.OBSERVATION, detector=detector
        )
        label = configs.CACHE.files(observation_id, scan)[".lbl"]
        name = Path(labels.load(label)[configs.WAVELENGTH_KEY]).stem
        archive.collect(
            client,
            name,
            configs.CACHE.files(configs.WAVELENGTH_DIR, name.lower()),
            pt=WAVELENGTH_TYPE,
            **ODE,
        )
