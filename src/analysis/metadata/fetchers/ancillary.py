"""A sample of every ancillary table still unread, fetched, read and dropped."""

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
from building.configs.sharad import NAMING
from building.download.archive import bring, published
from common import console as printing
from common.fetch.http import TLS_CONTEXT
from common.fetch.ode import ODEClient
from common.maths.tessellate import Tessellate
from common.pds import labels

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
    """Read a sample of the table of every product wanted over each of its tiles.

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
    with ODEClient() as ode_client:
        params = product_params(instrument_set, ancillary.pt)
        for item in every_product(ode_client, params, "pf"):
            kbytes = published(item, "KBytes")
            for name, url in published(item).items():
                if name.endswith(".tab"):
                    size = int(float(kbytes[name] or 0))
                    tables[NAMING.parse(item["pdsid"])] = (url, size)
                elif name.endswith(".lbl"):
                    label_url = url
    distortions: list[Distortion] = []
    failed = 0
    with (
        httpx.Client(verify=TLS_CONTEXT) as client,
        tempfile.TemporaryDirectory() as scratch_dir,
        ThreadPoolExecutor(max_workers=workers) as pool,
    ):
        scratch = Path(scratch_dir)
        layout = scratch / "layout.lbl"
        bring({".lbl": layout}, {".lbl": label_url}, client=client)
        asked = {
            ancillary.latitude,
            ancillary.longitude,
            ancillary.solar_zenith,
            ancillary.distortion,
        }
        columns = [
            column for column in labels.columns(layout) if column["NAME"] in asked
        ]
        label = labels.load(layout)
        futures = {
            pool.submit(
                sample_distortions,
                client,
                scratch,
                tables,
                label,
                columns,
                ancillary,
                grid,
                pdsid,
                groups,
            ): pdsid
            for pdsid, groups in wanted.items()
        }
        for done, future in enumerate(as_completed(futures), 1):
            pdsid = futures[future]
            console.print_progress(ancillary.pt.lower(), done, len(futures))
            try:
                distortions.extend(future.result())
            except Exception as error:
                failed += 1
                printing.print_failure(pdsid, error, failed)
    return distortions, failed


def sample_distortions(
    client: httpx.Client,
    scratch: Path,
    tables: Mapping[str, tuple[str, int]],
    label: dict[str, str],
    columns: list[dict[str, str]],
    ancillary: Ancillary,
    grid: Tessellate,
    pdsid: str,
    groups: dict[str, TileGroup],
) -> list[Distortion]:
    """Bring a sample of one product's table down, read it, and throw it away.

    Args:
        client: The client the byte ranges are asked over.
        scratch: The directory the sample is written to while it is read.
        tables: The URL and size in kilobytes of every table, by product name.
        label: The parsed label every table of the ancillary shares.
        columns: The COLUMN objects of the columns the ancillary reads alone.
        ancillary: What the ancillary is, and which of its columns are read.
        grid: The grid the tiles are cut from.
        pdsid: The product whose table is sampled.
        groups: The groups the product still has to be read over, by name.

    Returns:
        distortions: One per tile of those groups the sampled rows fall on.
    """
    table = scratch / f"{pdsid}.tab"
    row_bytes = int(label["ROW_BYTES"])
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
        rows = [body[at.end() : at.end() + row_bytes] for at in PART.finditer(body)]
        sampled.extend(rows or [body])
    table.write_bytes(b"".join(sampled))
    try:
        return load_distortions(table, pdsid, label, columns, ancillary, groups, grid)
    finally:
        table.unlink(missing_ok=True)
