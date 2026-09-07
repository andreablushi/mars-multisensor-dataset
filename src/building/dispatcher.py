"""Handing each stage of a build to the instrument whose product it is."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
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
from building.preprocessing.crism import preprocess as crism
from building.preprocessing.ctx import preprocess as ctx
from building.preprocessing.mola import preprocess as mola
from building.preprocessing.sharad import altitude
from building.preprocessing.sharad import preprocess as sharad

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
        discard: What deletes the product from the cache once it is built, and
            None for an archive small enough to be held for the whole run,
            which is what a product shared by many features asks for.
        observation_id: What reads which observation a product the selection
            kept belongs to, or None for an instrument the selection can never
            name.
        identifiers: What asks an archive which of its products hold one
            feature's ground, for the instrument the selection cannot name, and
            None for every instrument named by a product id.
        altitude: What reads how high the spacecraft flew, for a sounder whose
            delay axis is read through it, and None for every other instrument.
        worker_bytes: What one build holds of its largest product at once, the
            product itself and the crop and masks that stand beside it, which
            is what a build is given where nothing measures the product itself.
        held_bytes: What reads how much one downloaded product holds, off the
            files it landed as, for an instrument whose products differ in size
            by more than the memory a run can spare, and None for the rest.
    """

    layout: Layout
    fetch: Callable[[str, httpx.Client], None]
    read_observation: Callable[[str], Any]
    crop: Callable[..., Any]
    discard: Callable[[str], None] | None = None
    observation_id: Callable[[str], str | None] | None = None
    identifiers: Callable[[Feature, httpx.Client], list[str]] | None = None
    altitude: Callable[[Any], tuple[float, float]] | None = None
    worker_bytes: int = 512 * 1024**2
    held_bytes: Callable[[str], int] | None = None

    def holds(self, identifier: str) -> int:
        """Return how much memory one build of one product of this holds.

        Args:
            identifier: The product, already downloaded, since what it holds is
                measured off the files it landed as.

        Returns:
            How many bytes to hold for it, which is what it was declared to
            hold where its own size is not read.
        """
        return self.held_bytes(identifier) if self.held_bytes else self.worker_bytes


INSTRUMENTS = {
    crism_configs.LAYOUT.instrument: Instrument(
        crism_configs.LAYOUT,
        crism_download.fetch,
        crism.read_observation,
        crism.crop,
        discard=crism_configs.CACHE.discard,
        observation_id=crism_configs.NAMING.parse,
        # A cleaned observation measured 203 MB, both detectors and the chain.
        worker_bytes=512 * 1024**2,
    ),
    ctx_configs.LAYOUT.instrument: Instrument(
        ctx_configs.LAYOUT,
        ctx_download.fetch,
        ctx.read_observation,
        ctx.crop,
        discard=ctx_configs.CACHE.discard,
        observation_id=ctx_configs.NAMING.parse,
        # A 2.6 GB scan, held with its crop and two masks; 829 MB measured 2.0 GB.
        worker_bytes=9 * 1024**3,
        held_bytes=ctx.held_bytes,
    ),
    mola_configs.LAYOUT.instrument: Instrument(
        mola_configs.LAYOUT,
        mola_download.fetch,
        mola.read_observation,
        mola.crop,
        identifiers=mola_download.tiles,
        # The whole gridded record is 2 GB, so a tile is held for the run.
        worker_bytes=256 * 1024**2,
    ),
    sharad_configs.LAYOUT.instrument: Instrument(
        sharad_configs.LAYOUT,
        sharad_download.fetch,
        sharad.read_observation,
        sharad.crop,
        discard=sharad_configs.CACHE.discard,
        observation_id=sharad_configs.NAMING.parse,
        altitude=altitude.altitude_m,
        # A radargram and its geometry measured 135 MB.
        worker_bytes=256 * 1024**2,
    ),
}
