"""The distortion of every look over each tile, gathered into one file and read back."""

from __future__ import annotations

import functools

import pyarrow.parquet as pq

from analysis import paths
from analysis.metadata.fetchers.ancillary import fetch_distortions
from analysis.models.ancillary import Distortion
from analysis.models.instrument import InstrumentSet
from analysis.models.settings import Settings
from analysis.models.tile_group import TileGroup
from analysis.utils.tile_group import every_tile_group, tile_grid
from common.disk import parquet
from common.disk.files import read_jsonl

DISTORTIONS = parquet.schema_of(Distortion)


def summarise_ancillary(settings: Settings, force: bool = False) -> int:
    """Read every look not yet read, and write its distortion into the summary.

    Args:
        settings: The settled choices for the run, naming each set's ancillary.
        force: Whether to read every look again rather than trust the summary.

    Returns:
        failed: How many tables could not be read, left to the next run.
    """
    summarised = [] if force else list(read_distortions())
    done = {(known.group, known.pdsid) for known in summarised}
    grid = tile_grid()
    groups = {
        group.name: group for group in every_tile_group(grid, settings.tile_group_deg)
    }
    sets = paths.metadata_files()
    failed = 0
    for key, ancillary in settings.ancillary.items():
        instrument_set = InstrumentSet.from_key(key)
        wanted: dict[str, dict[str, TileGroup]] = {}
        for source in (path for path in sets if path.stem == instrument_set.slug):
            name = source.parent.name
            for record in read_jsonl(source):
                if (name, record["pdsid"]) not in done:
                    wanted.setdefault(record["pdsid"], {})[name] = groups[name]
        if not wanted:
            continue
        distortions, lost = fetch_distortions(
            wanted, instrument_set, ancillary, grid, settings.workers
        )
        failed += lost
        summarised.extend(distortions)
    parquet.write(summarised, DISTORTIONS, paths.DISTORTIONS_PATH)
    read_distortions.cache_clear()
    return failed


@functools.lru_cache(maxsize=1)
def read_distortions(group: str | None = None) -> tuple[Distortion, ...]:
    """Read the distortion of every look over each tile of one group, cached.

    Args:
        group: The name of the tile group, or None for every group.

    Returns:
        distortions: Each look's distortion over each tile, in the order written.
    """
    if not paths.DISTORTIONS_PATH.exists():
        return ()
    filters = None if group is None else [("group", "==", group)]
    rows = pq.read_table(paths.DISTORTIONS_PATH, schema=DISTORTIONS, filters=filters)
    return tuple(Distortion(**row) for row in rows.to_pylist())
