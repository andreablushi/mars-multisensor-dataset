"""One product fetched, read and cut to a tile, the way a training build does it."""

from __future__ import annotations

from typing import Any

import httpx

from building.models.instrument import INSTRUMENTS
from building.preprocessing.common.models.sample import Sample
from common.config import training_settings
from common.fetch.http import TLS_CONTEXT
from common.models.tile import Tile


def fetch_product(name: str, identifier: str, frame: Tile) -> None:
    """Bring one product into the build's cache and place it, unless it is there.

    Args:
        name: The instrument, as the build names it.
        identifier: What that instrument is asked for, its observation.
        frame: The tile it is fetched for.
    """
    instrument = INSTRUMENTS[name]
    with httpx.Client(verify=TLS_CONTEXT) as client:
        instrument.fetch(identifier, client, (frame,))
    if instrument.place:
        instrument.place(identifier)


def product_observation(name: str, identifier: str) -> Any:
    """Read one fetched product with the settings a training build hands it."""
    preprocessing = training_settings().preprocessing.get(name, {})
    return INSTRUMENTS[name].read_observation(identifier, **preprocessing)


def tile_sample(name: str, observation: Any, frame: Tile) -> Sample:
    """Cut a read product to a tile, as a training build does.

    Args:
        name: The instrument, as the build names it.
        observation: The product as its reader handed it back.
        frame: The tile it is cut to.

    Returns:
        sample: The crop the build would write.

    Raises:
        ValueError: When the product measured nothing or reaches none of the tile.
    """
    sample = None if observation is None else INSTRUMENTS[name].crop(observation, frame)
    if sample is None:
        raise ValueError(f"{name} holds nothing of tile {frame.name}.")
    return sample
