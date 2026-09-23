"""Summarising every companion table into the records of the set it sits beside."""

from __future__ import annotations

from analysis import paths
from analysis.metadata import file_explorer
from analysis.metadata.companions import fetch, write
from analysis.metadata.companions.summarise import Ledger
from analysis.metadata.fetchers.companions import stem_of
from analysis.models.instrument import InstrumentSet
from analysis.models.settings import Settings
from analysis.models.tile_group import TileGroup
from analysis.utils import tile_group
from common.disk.files import read_jsonl
from common.maths.tessellate import Tessellate


def annotate_companions(settings: Settings, force: bool = False) -> int:
    """Summarise every companion table not yet read, and write it into its records.

    Args:
        settings: The settled choices for the run, naming each set's companion.
        force: Whether to read every table again rather than trust the ledger.

    Returns:
        failed: How many tables could not be summarised, left to the next run.
    """
    grid = Tessellate.of(settings.tile_km)
    groups = {
        group.name: group
        for group in tile_group.every_tile_group(grid, settings.tile_group_deg)
    }
    stored = file_explorer.find_sets()
    failed = 0
    for key, companion in settings.companions.items():
        instrument_set = InstrumentSet.from_key(key)
        named = [f'"{field}"' for field in companion.columns]
        unsettled = [
            path
            for path in stored
            if path.stem == instrument_set.slug
            and (
                force
                or any(
                    field not in line
                    for line in path.read_text(encoding="utf-8").splitlines()
                    for field in named
                )
            )
        ]
        ledger_path = paths.companion_ledger(
            paths.METADATA_ROOT, instrument_set, companion.pt
        )
        if force:
            ledger_path.unlink(missing_ok=True)
        ledger: Ledger = {}
        if ledger_path.exists():
            for line in read_jsonl(ledger_path):
                ledger.setdefault(line["product"], {}).update(line["groups"])
        wanted: dict[str, dict[str, TileGroup]] = {}
        for path in unsettled:
            name = path.parent.name
            for record in read_jsonl(path):
                stem = stem_of(record["pdsid"])
                if name not in ledger.get(stem, {}):
                    wanted.setdefault(stem, {})[name] = groups[name]
        if wanted:
            failed += fetch.fetch_tables(
                wanted, instrument_set, companion, settings.workers, ledger, ledger_path
            )
        write.write_companions(unsettled, ledger, companion)
    return failed
