"""Fetching every ancillary table still unread, reading it, and dropping it."""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

from analysis import console
from analysis.metadata.fetchers.products import every_product, product_params
from analysis.metadata.loaders.ancillary import load_distortions
from analysis.models.ancillary import Ancillary, Distortion
from analysis.models.instrument import InstrumentSet
from analysis.models.tile_group import TileGroup
from building.common.pds import labels
from building.configs.sharad import NAMING
from building.download.archive import bring, published
from common import console as printing
from common.fetch.ode import ODEClient


def fetch_distortions(
    wanted: Mapping[str, dict[str, TileGroup]],
    instrument_set: InstrumentSet,
    ancillary: Ancillary,
    workers: int,
) -> tuple[list[Distortion], int]:
    """Read the table of every product wanted over each group it is wanted in.

    Args:
        wanted: The groups each product still has to be read over, by pdsid.
        instrument_set: The set the ancillary is published beside.
        ancillary: What the ancillary is, and which of its columns are read.
        workers: How many tables are brought down at once.

    Returns:
        distortions: One per product and group it was wanted in.
        failed: How many tables could not be read, left to the next run.
    """
    tables: dict[str, str] = {}
    label_url = ""
    with ODEClient() as asking:
        params = product_params(instrument_set, ancillary.pt)
        for item in every_product(asking, params, "pf"):
            for name, url in published(item).items():
                if name.endswith(".tab"):
                    tables[NAMING.parse(item["pdsid"])] = url
                elif name.endswith(".lbl"):
                    label_url = url
    distortions: list[Distortion] = []
    failed = 0
    progress = console.logged(ancillary.pt.lower())
    with (
        httpx.Client() as client,
        tempfile.TemporaryDirectory() as held,
        ThreadPoolExecutor(max_workers=workers) as pool,
    ):
        scratch = Path(held)
        layout = scratch / "layout.lbl"
        bring({".lbl": layout}, {".lbl": label_url}, client=client)
        asked = {
            ancillary.latitude,
            ancillary.longitude,
            ancillary.solar_zenith,
            ancillary.distortion,
        }
        columns = [one for one in labels.columns(layout) if one["NAME"] in asked]
        label = labels.load(layout)

        def table_distortions(
            pdsid: str, groups: dict[str, TileGroup]
        ) -> list[Distortion]:
            """Bring one table down, read it, and throw it away."""
            table = scratch / f"{pdsid}.tab"
            bring(
                {".tab": table},
                {".tab": tables.get(NAMING.parse(pdsid), "")},
                client=client,
            )
            try:
                return load_distortions(table, pdsid, label, columns, ancillary, groups)
            finally:
                table.unlink(missing_ok=True)

        futures = {
            pool.submit(table_distortions, pdsid, groups): pdsid
            for pdsid, groups in wanted.items()
        }
        for done, future in enumerate(as_completed(futures), 1):
            pdsid = futures[future]
            progress(done, len(futures))
            try:
                distortions.extend(future.result())
            except Exception as error:
                failed += 1
                printing.named_failure(pdsid, error, failed)
    return distortions, failed
