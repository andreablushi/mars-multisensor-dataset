"""The distortion of every SHARAD look over each tile, gathered and read back."""

from __future__ import annotations

import functools

from analysis import paths
from analysis.metadata.fetchers.ancillary import fetch_distortions
from analysis.models.ancillary import Distortion
from analysis.models.settings import AnalysisSettings
from analysis.models.tile_group import TileGroup
from analysis.utils.tile_group import every_tile_group, tile_grid
from building.configs import sharad
from common.disk import parquet
from common.disk.files import read_jsonl

DISTORTIONS = parquet.schema_of(Distortion)


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
        if instrument_set.iid != sharad.LAYOUT.instrument:
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
    return tuple(
        parquet.read_rows(Distortion, DISTORTIONS, paths.DISTORTIONS_PATH, filters)
    )
