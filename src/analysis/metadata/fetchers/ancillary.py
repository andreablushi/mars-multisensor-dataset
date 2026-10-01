"""A sample of every SHARAD geometry table still unread, fetched, read and written."""

from __future__ import annotations

import re
import tempfile
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

from analysis import console, paths
from analysis.metadata.fetchers.products import every_product, product_params
from analysis.metadata.loaders.ancillary import (
    DISTORTIONS,
    load_distortions,
    read_distortions,
)
from analysis.models.ancillary import Distortion
from analysis.models.instrument import InstrumentSet
from analysis.models.settings import AnalysisSettings
from analysis.models.tile_group import TileGroup
from analysis.partition import Tessellate
from analysis.utils.tile_group import every_tile_group, tile_grid
from building.configs.sharad import LAYOUT, NAMING, PRODUCT_TYPES, Kind
from building.download.archive import download_files, file_fields
from building.preprocessing.sharad.models.observation import (
    LATITUDE_FIELD,
    LONGITUDE_FIELD,
    SOLAR_ZENITH_FIELD,
)
from common import console as printing
from common.disk import parquet
from common.disk.files import read_jsonl
from common.fetch.http import TLS_CONTEXT
from common.pds import labels
from common.pool import cancellable_pool

SAMPLED_EVERY = 8

RANGES_PER_REQUEST = 500

PART = re.compile(rb"Content-Range: bytes \d+-\d+/\d+\r\n\r\n")

GEOMETRY = PRODUCT_TYPES[Kind.GEOMETRY]


def write_distortions(settings: AnalysisSettings, force: bool) -> int:
    """Read every SHARAD look not yet read, and write its distortion out.

    Args:
        settings: The settled choices for the run, naming the SHARAD sets.
        force: Whether to read every look again rather than trust those written.

    Returns:
        failed: How many tables could not be read, left to the next run.
    """
    gathered = [] if force else list(read_distortions())
    done = {(known.group, known.pdsid) for known in gathered}
    grid = tile_grid()
    groups = {
        group.name: group for group in every_tile_group(grid, settings.tile_group_deg)
    }
    sets = paths.metadata_files()
    failed = 0
    for instrument_set in settings.instrument_sets:
        if instrument_set.iid != LAYOUT.instrument:
            continue
        wanted: dict[str, dict[str, TileGroup]] = {}
        for source in (path for path in sets if path.stem == instrument_set.slug):
            name = source.parent.name
            for record in read_jsonl(source):
                if (name, record["pdsid"]) not in done:
                    wanted.setdefault(record["pdsid"], {})[name] = groups[name]
        if not wanted:
            continue
        distortions, lost = fetch_distortions(wanted, instrument_set, settings, grid)
        failed += lost
        gathered.extend(distortions)
    parquet.write_rows(gathered, DISTORTIONS, paths.DISTORTIONS_PATH)
    read_distortions.cache_clear()
    return failed


def fetch_distortions(
    wanted: Mapping[str, dict[str, TileGroup]],
    instrument_set: InstrumentSet,
    settings: AnalysisSettings,
    grid: Tessellate,
) -> tuple[list[Distortion], int]:
    """Read a sample of the geometry table of every SHARAD track wanted over its tiles.

    Args:
        wanted: The groups each product still has to be read over, by pdsid.
        instrument_set: The SHARAD set the geometry is published beside.
        settings: The settled choices for the run, naming the distortion column,
            the night threshold and how many tables are brought down at once.
        grid: The grid the tiles are cut from.

    Returns:
        distortions: One per product and tile its rows fall on.
        failed: How many tables could not be read, left to the next run.
    """
    tables: dict[str, tuple[str, int]] = {}
    label_url = ""
    distortions: list[Distortion] = []
    failed = 0
    with (
        httpx.Client(verify=TLS_CONTEXT) as client,
        tempfile.TemporaryDirectory() as scratch_dir,
        cancellable_pool(ThreadPoolExecutor(settings.workers)) as pool,
    ):
        params = product_params(instrument_set, GEOMETRY)
        for item in every_product(client, params, "pf"):
            kbytes = file_fields(item, "KBytes")
            for name, url in file_fields(item).items():
                if name.endswith(".tab"):
                    size = int(float(kbytes[name] or 0))
                    tables[NAMING.observation_id(item["pdsid"])] = (url, size)
                elif name.endswith(".lbl"):
                    label_url = url
        scratch = Path(scratch_dir)
        layout = scratch / "layout.lbl"
        download_files({".lbl": layout}, {".lbl": label_url}, client=client)
        asked = {
            LATITUDE_FIELD,
            LONGITUDE_FIELD,
            SOLAR_ZENITH_FIELD,
            settings.sharad_distortion,
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
                settings,
                grid,
                pdsid,
                groups,
            ): pdsid
            for pdsid, groups in wanted.items()
        }
        for done, future in enumerate(as_completed(futures), 1):
            pdsid = futures[future]
            console.print_progress(GEOMETRY.lower(), done, len(futures))
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
    settings: AnalysisSettings,
    grid: Tessellate,
    pdsid: str,
    groups: dict[str, TileGroup],
) -> list[Distortion]:
    """Bring a sample of one product's table down, read it, and throw it away.

    Args:
        client: The client the byte ranges are asked over.
        scratch: The directory the sample is written to while it is read.
        tables: The URL and size in kilobytes of every table, by product name.
        label: The parsed label every geometry table shares.
        columns: The COLUMN objects of the columns read alone.
        settings: The settled choices for the run, naming the distortion column
            and the night threshold.
        grid: The grid the tiles are cut from.
        pdsid: The product whose table is sampled.
        groups: The groups the product still has to be read over, by name.

    Returns:
        distortions: One per tile of those groups the sampled rows fall on.
    """
    table = scratch / f"{pdsid}.tab"
    row_bytes = int(label["ROW_BYTES"])
    url, kbytes = tables.get(NAMING.observation_id(pdsid), ("", 0))
    end = (kbytes - 1) * 1024 - row_bytes + 1
    starts = range(0, max(end, 0), row_bytes * SAMPLED_EVERY)
    chunks = [
        starts[at : at + RANGES_PER_REQUEST]
        for at in range(0, len(starts), RANGES_PER_REQUEST)
    ]
    sampled: list[bytes] = []
    for chunk in chunks or [range(0)]:
        spans = tuple((at, at + row_bytes) for at in chunk)
        download_files({".tab": table}, {".tab": url}, client=client, spans=spans)
        body = table.read_bytes()
        table.unlink()
        rows = [body[at.end() : at.end() + row_bytes] for at in PART.finditer(body)]
        sampled.extend(rows or [body])
    table.write_bytes(b"".join(sampled))
    try:
        return load_distortions(table, pdsid, label, columns, settings, groups, grid)
    finally:
        table.unlink(missing_ok=True)
