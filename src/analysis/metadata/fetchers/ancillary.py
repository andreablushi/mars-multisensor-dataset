"""Fetching a sample of every ancillary table still unread, read and then dropped."""

from __future__ import annotations

import re
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
from common.maths.tessellate import Tessellate

SAMPLED_EVERY = 8

RANGES_PER_REQUEST = 500

PART = re.compile(rb"Content-Range: bytes \d+-\d+/\d+\r\n\r\n")


def fetch_distortions(
    wanted: Mapping[str, dict[str, TileGroup]],
    instrument_set: InstrumentSet,
    ancillary: Ancillary,
    grid: Tessellate,
    workers: int,
) -> tuple[list[Distortion], int]:
    """Read a sample of the table of every product wanted over each tile asking.

    Args:
        wanted: The groups each product still has to be read over, by pdsid.
        instrument_set: The set the ancillary is published beside.
        ancillary: What the ancillary is, and which of its columns are read.
        grid: The grid the tiles are cut from.
        workers: How many tables are brought down at once.

    Returns:
        distortions: One per product and tile its rows fall on.
        failed: How many tables could not be read, left to the next run.
    """
    tables: dict[str, tuple[str, int]] = {}
    label_url = ""
    with ODEClient() as asking:
        params = product_params(instrument_set, ancillary.pt)
        for item in every_product(asking, params, "pf"):
            kbytes = published(item, "KBytes")
            for name, url in published(item).items():
                if name.endswith(".tab"):
                    size = int(float(kbytes[name] or 0))
                    tables[NAMING.parse(item["pdsid"])] = (url, size)
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
        row_bytes = int(label["ROW_BYTES"])

        def table_distortions(
            pdsid: str, groups: dict[str, TileGroup]
        ) -> list[Distortion]:
            """Bring a sample of one table down, read it, and throw it away."""
            table = scratch / f"{pdsid}.tab"
            url, kbytes = tables.get(NAMING.parse(pdsid), ("", 0))
            end = (kbytes - 1) * 1024 - row_bytes + 1
            starts = range(0, max(end, 0), row_bytes * SAMPLED_EVERY)
            chunks = [
                starts[at : at + RANGES_PER_REQUEST]
                for at in range(0, len(starts), RANGES_PER_REQUEST)
            ]
            sampled: list[bytes] = []
            for chunk in chunks or [range(0)]:
                spans = tuple((at, at + row_bytes) for at in chunk)
                bring({".tab": table}, {".tab": url}, client=client, spans=spans)
                body = table.read_bytes()
                table.unlink()
                rows = [
                    body[at.end() : at.end() + row_bytes] for at in PART.finditer(body)
                ]
                sampled.extend(rows or [body])
            table.write_bytes(b"".join(sampled))
            try:
                return load_distortions(
                    table, pdsid, label, columns, ancillary, groups, grid
                )
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
