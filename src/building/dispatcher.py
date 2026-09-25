"""Handing each stage of a build to the instrument whose product it is."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import httpx

from building.common.layout import Layout
from building.configs import crism as crism_configs
from building.configs import ctx as ctx_configs
from building.configs import mola as mola_configs
from building.configs import sharad as sharad_configs
from building.download import crism as crism_download
from building.download import ctx as ctx_download
from building.download import mola as mola_download
from building.download import sharad as sharad_download
from building.preprocessing.crism import preprocess as crism
from building.preprocessing.ctx import preprocess as ctx
from building.preprocessing.mola import preprocess as mola
from building.preprocessing.sharad import preprocess as sharad
from common.models.tile import Tile


class Archive(StrEnum):
    """The lanes downloads wait on: one per archive, and the SPICE server."""

    JPL = "jpl"
    WUSTL = "wustl"
    SPICE = "spice"


@dataclass(frozen=True, slots=True)
class Instrument:
    """How one instrument goes from a product ODE holds to a crop in the dataset.

    Attributes:
        layout: What its arrays hold, and which of them it is stored for.
        fetch: What brings one product down into the cache, as much as its tiles need.
        read_observation: What reads a fetched product off disk.
        crop: What cuts that observation to a tile's box, or None where it misses.
        archive: Which archive its products are downloaded from.
        discard: What deletes the built product, or None for a small archive.
        place: What readies a fetched product on the SPICE server, or None.
        observation_id: What reads a kept product's observation, or None.
        identifiers: What asks an archive what covers a tile, or None.
        worker_bytes: What one build holds of its largest product at once.
    """

    layout: Layout
    fetch: Callable[[str, httpx.Client, tuple[Tile, ...]], None]
    read_observation: Callable[[str], Any]
    crop: Callable[..., Any]
    archive: Archive
    discard: Callable[[str], None] | None = None
    place: Callable[[str], None] | None = None
    observation_id: Callable[[str], str | None] | None = None
    identifiers: Callable[[Tile, httpx.Client], list[str]] | None = None
    worker_bytes: int = 512 * 1024**2


INSTRUMENTS = {
    crism_configs.LAYOUT.instrument: Instrument(
        crism_configs.LAYOUT,
        crism_download.fetch,
        crism.read_observation,
        crism.crop,
        Archive.WUSTL,
        discard=crism_configs.CACHE.discard,
        observation_id=crism_configs.NAMING.parse,
        # A hyperspectral observation takes 601 MB against 325 MB, so it goes first.
        worker_bytes=1024**3,
    ),
    ctx_configs.LAYOUT.instrument: Instrument(
        ctx_configs.LAYOUT,
        ctx_download.fetch,
        ctx.read_observation,
        ctx.crop,
        Archive.JPL,
        discard=ctx_configs.CACHE.discard,
        place=ctx_download.place,
        observation_id=ctx_configs.NAMING.parse,
        # A tile's window, its crop and two masks, beside ISIS measured at 513 MB.
        worker_bytes=2 * 1024**3,
    ),
    mola_configs.LAYOUT.instrument: Instrument(
        mola_configs.LAYOUT,
        mola_download.fetch,
        mola.read_observation,
        mola.crop,
        Archive.WUSTL,
        identifiers=mola_download.tile_grids,
        # The whole gridded record is 2 GB, so a sheet is held for the run.
        worker_bytes=256 * 1024**2,
    ),
    sharad_configs.LAYOUT.instrument: Instrument(
        sharad_configs.LAYOUT,
        sharad_download.fetch,
        sharad.read_observation,
        sharad.crop,
        Archive.WUSTL,
        discard=sharad_configs.CACHE.discard,
        observation_id=sharad_configs.NAMING.parse,
        # A radargram, its geometry and its clutter measured 222 MB at peak.
        worker_bytes=256 * 1024**2,
    ),
}
