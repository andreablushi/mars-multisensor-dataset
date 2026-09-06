"""Handing each stage of a build to the instrument whose product it is."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
from building.preprocessing.crism import crop as crism_crop
from building.preprocessing.crism import read as crism_read
from building.preprocessing.crism import write as crism_write
from building.preprocessing.ctx import crop as ctx_crop
from building.preprocessing.ctx import read as ctx_read
from building.preprocessing.ctx import write as ctx_write
from building.preprocessing.mola import crop as mola_crop
from building.preprocessing.mola import read as mola_read
from building.preprocessing.mola import write as mola_write
from building.preprocessing.sharad import altitude
from building.preprocessing.sharad import crop as sharad_crop
from building.preprocessing.sharad import read as sharad_read
from building.preprocessing.sharad import write as sharad_write

if TYPE_CHECKING:
    from analysis.models.feature import Feature


@dataclass(frozen=True, slots=True)
class Instrument:
    """How one instrument goes from a product ODE holds to a crop in the dataset.

    Attributes:
        layout: What its arrays hold, and which of them it is stored for.
        fetch: What brings one product of it down into the cache.
        read_observation: What reads a fetched product off disk, whole.
        crop: What cuts that observation to one feature's box, handing back the
            sample to store or None where it reaches none of it.
        write: What writes the sample into the dataset.
        observation_id: What reads which observation a product the selection
            kept belongs to, or None for an instrument the selection can never
            name.
        identifiers: What asks an archive which of its products hold one
            feature's ground, for the instrument the selection cannot name, and
            None for every instrument named by a product id.
        altitude: What reads how high the spacecraft flew, for a sounder whose
            delay axis is read through it, and None for every other instrument.
    """

    layout: Layout
    fetch: Callable[[str, httpx.Client], None]
    read_observation: Callable[[str], Any]
    crop: Callable[..., Any]
    write: Callable[..., Path]
    observation_id: Callable[[str], str | None] | None = None
    identifiers: Callable[[Feature, httpx.Client], list[str]] | None = None
    altitude: Callable[[Any], tuple[float, float]] | None = None


INSTRUMENTS = {
    crism_configs.LAYOUT.instrument: Instrument(
        crism_configs.LAYOUT,
        crism_download.fetch,
        crism_read.read_observation,
        crism_crop.crop,
        crism_write.write,
        observation_id=crism_configs.NAMING.parse,
    ),
    ctx_configs.LAYOUT.instrument: Instrument(
        ctx_configs.LAYOUT,
        ctx_download.fetch,
        ctx_read.read_observation,
        ctx_crop.crop,
        ctx_write.write,
        observation_id=ctx_configs.NAMING.parse,
    ),
    mola_configs.LAYOUT.instrument: Instrument(
        mola_configs.LAYOUT,
        mola_download.fetch,
        mola_read.read_observation,
        mola_crop.crop,
        mola_write.write,
        identifiers=mola_download.tiles,
    ),
    sharad_configs.LAYOUT.instrument: Instrument(
        sharad_configs.LAYOUT,
        sharad_download.fetch,
        sharad_read.read_observation,
        sharad_crop.crop,
        sharad_write.write,
        observation_id=sharad_configs.NAMING.parse,
        altitude=altitude.altitude_m,
    ),
}
