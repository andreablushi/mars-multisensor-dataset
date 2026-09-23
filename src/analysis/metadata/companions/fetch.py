"""Bringing every companion table still unread down, summarising it, and dropping it."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

from analysis import console
from analysis.metadata.companions.summarise import Ledger, Summary, summarise_table
from analysis.metadata.fetchers.companions import companion_urls
from analysis.metadata.ode import ODEClient
from analysis.models.companion import Companion
from analysis.models.instrument import InstrumentSet
from analysis.models.tile_group import TileGroup
from building.common.pds import labels
from common import console as printing
from common.fetch import http

TIMEOUT = 120.0


def fetch_tables(
    wanted: Mapping[str, dict[str, TileGroup]],
    instrument_set: InstrumentSet,
    companion: Companion,
    workers: int,
    ledger: Ledger,
    ledger_path: Path,
) -> int:
    """Summarise the table of every product wanted, writing each down as it lands.

    Args:
        wanted: The groups each product still has to be summarised over, by stem.
        instrument_set: The set the companion is published beside.
        companion: What the companion is, and which of its columns are asked for.
        workers: How many tables are brought down at once.
        ledger: What every table was summarised as so far, added to in place.
        ledger_path: Where each summary is appended the moment it lands.

    Returns:
        failed: How many tables could not be summarised, left to the next run.
    """
    with ODEClient() as asking:
        urls = companion_urls(asking, instrument_set, companion.pt)
    failed = 0
    progress = console.logged(companion.pt.lower())
    with (
        httpx.Client() as client,
        tempfile.TemporaryDirectory() as held,
        ThreadPoolExecutor(max_workers=workers) as pool,
        ledger_path.open("a", encoding="utf-8") as handle,
    ):
        scratch = Path(held)
        # Every table of one product type is laid out alike, so one label says it all
        label = scratch / "layout.lbl"
        http.streamed(
            next(one[".lbl"] for one in urls.values() if ".lbl" in one),
            label,
            TIMEOUT,
            client=client,
        )
        asked = {companion.latitude, companion.longitude, *companion.columns.values()}
        columns = [one for one in labels.columns(label) if one["NAME"] in asked]
        row_bytes = int(labels.load(label)["ROW_BYTES"])

        def summarised(stem: str, groups: dict[str, TileGroup]) -> dict[str, Summary]:
            """Bring one table down, summarise it, and throw it away."""
            if ".tab" not in urls.get(stem, {}):
                raise FileNotFoundError(f"ODE offers no {companion.pt} for {stem}")
            table = scratch / f"{stem}.tab"
            http.streamed(urls[stem][".tab"], table, TIMEOUT, client=client)
            try:
                return summarise_table(table, columns, row_bytes, companion, groups)
            finally:
                table.unlink(missing_ok=True)

        futures = {
            pool.submit(summarised, stem, groups): stem
            for stem, groups in wanted.items()
        }
        for done, future in enumerate(as_completed(futures), 1):
            stem = futures[future]
            progress(done, len(futures))
            try:
                found = future.result()
            except Exception as error:
                failed += 1
                printing.named_failure(stem, error, failed)
                continue
            ledger.setdefault(stem, {}).update(found)
            handle.write(json.dumps({"product": stem, "groups": found}) + "\n")
            handle.flush()
    return failed
