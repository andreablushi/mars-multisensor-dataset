"""Writing the Sun and the ionosphere over every SHARAD track into its records."""

from __future__ import annotations

import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

from analysis import console, paths
from analysis.metadata.ode import ODEClient
from analysis.metadata.sounding import ledger as ledgers
from analysis.metadata.sounding import locate
from analysis.metadata.sounding.measure import (
    PHASE_FIELD,
    SOLAR_ZENITH_FIELD,
    Measured,
    measure_track,
)
from analysis.models.settings import Settings
from analysis.utils import tile_group
from common import console as printing
from common.disk.files import read_jsonl, write_jsonl
from common.fetch import http
from common.maths.tessellate import Tessellate

SOUNDER = "SHARAD"

# How long one geometry table may pause between chunks, in seconds
TIMEOUT = 120.0


def annotate_soundings(settings: Settings, force: bool = False) -> int:
    """Measure every SHARAD track not yet measured, and write it into its records.

    Args:
        settings: The settled choices for the run, which size the pool.
        force: Whether to measure every track again rather than read the ledger.

    Returns:
        failed: How many tracks could not be measured, left to the next run.
    """
    grid = Tessellate.of(settings.tile_km)
    groups = {
        group.name: group
        for group in tile_group.every_tile_group(grid, settings.tile_group_deg)
    }
    sounders = [one for one in settings.instrument_sets if one.iid == SOUNDER]
    files = [
        held
        for sounder in sounders
        for name in groups
        if (held := paths.metadata_file(paths.METADATA_ROOT, name, sounder)).exists()
        and held.stat().st_size
    ]
    listed: dict[str, set[str]] = {}
    # A file every record of which is already measured is left unwritten
    settled: set[Path] = set()
    for held in files:
        complete = True
        for record in read_jsonl(held):
            listed.setdefault(locate.track_of(record["pdsid"]), set()).add(
                held.parent.name
            )
            complete &= SOLAR_ZENITH_FIELD in record
        if complete and not force:
            settled.add(held)
    ledger = {} if force else ledgers.read_ledger()
    wanted = {
        track: names
        for track, names in listed.items()
        if not names <= ledger.get(track, {}).keys()
    }
    failed = 0
    if wanted:
        with ODEClient() as asking:
            urls = {
                track: url
                for sounder in sounders
                for track, url in locate.geometry_urls(asking, sounder).items()
            }

        def measured(
            track: str, names: set[str], client: httpx.Client, scratch: Path
        ) -> dict[str, Measured]:
            """Bring one track's geometry down, measure it, and throw it away."""
            if track not in urls:
                raise FileNotFoundError(f"ODE offers no geometry for {track}")
            table = scratch / f"{track}{locate.TABLE_SUFFIX}"
            http.streamed(urls[track], table, TIMEOUT, client=client)
            try:
                return measure_track(table, {name: groups[name] for name in names})
            finally:
                table.unlink(missing_ok=True)

        progress = console.logged("soundings")
        with (
            httpx.Client() as client,
            tempfile.TemporaryDirectory() as scratch,
            ThreadPoolExecutor(max_workers=settings.workers) as pool,
            paths.SOUNDINGS_PATH.open("w" if force else "a") as handle,
        ):
            futures = {
                pool.submit(measured, track, names, client, Path(scratch)): track
                for track, names in wanted.items()
            }
            for done, future in enumerate(as_completed(futures), 1):
                track = futures[future]
                progress(done, len(futures))
                try:
                    found = future.result()
                except Exception as error:
                    failed += 1
                    printing.named_failure(track, error, failed)
                    continue
                ledger.setdefault(track, {}).update(found)
                ledgers.append_ledger(handle, track, found)
    for held in files:
        if held in settled:
            continue
        group = held.parent.name
        records = list(read_jsonl(held))
        for record in records:
            track = ledger.get(locate.track_of(record["pdsid"]), {})
            if group in track:
                record[SOLAR_ZENITH_FIELD], record[PHASE_FIELD] = track[group] or (
                    None,
                    None,
                )
        write_jsonl(held, records)
    return failed
