"""The coverage artifacts a run left: gathering them into an index, and reading it."""

from __future__ import annotations

from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq

from common.analysis import configs, paths
from common.analysis.coverage.artifacts.write import EVENTS, SUMMARY
from common.analysis.coverage.models.coverage import Event, SetCoverage
from common.analysis.coverage.models.summary import Summary
from common.analysis.metadata import file_explorer
from common.analysis.utils import tile_group
from common.disk.files import atomic_path
from common.maths.tessellate import split_bands_columns
from common.models.tile import Tile


def reindex() -> int:
    """Rebuild the grid-wide summary from every group's summaries on disk.

    Returns:
        rows: How many summary rows the index holds.
    """
    found = sorted(paths.GROUPS_ROOT.glob(f"*/*{paths.SET_SUMMARY_SUFFIX}"))
    tables = [pq.read_table(path, schema=SUMMARY) for path in found]
    combined = pa.concat_tables(tables) if tables else SUMMARY.empty_table()
    with atomic_path(paths.catalog_summary_path()) as tmp:
        pq.write_table(combined, tmp, compression="zstd")
    return combined.num_rows


def catalogued_rows() -> list[Summary]:
    """Read every row the computed artifacts hold anywhere.

    Returns:
        rows: One row per tile and instrument set measured, in index order.
    """
    path = paths.catalog_summary_path()
    if not path.exists():
        return []
    return [Summary(**row) for row in pq.read_table(path, schema=SUMMARY).to_pylist()]


def measured_groups() -> list[str]:
    """Name every tile group holding a measured set on disk, busiest first.

    Returns:
        groups: The group names, the ones holding the most observation rows first.
    """
    return sorted(
        (
            directory.name
            for directory in paths.GROUPS_ROOT.glob("*")
            if any(directory.glob(f"*{paths.SET_SUMMARY_SUFFIX}"))
        ),
        key=lambda name: (
            -sum(
                path.stat().st_size
                for path in (paths.GROUPS_ROOT / name).glob(f"*{paths.EVENTS_SUFFIX}")
            )
        ),
    )


def load_tile(tile: Tile) -> list[SetCoverage]:
    """Read every instrument set for one tile, observed or not.

    Args:
        tile: The tile to read.

    Returns:
        coverage: One entry per instrument set, widest coverage first, then busiest,
            and nothing at all where no set reached it.
    """
    settings = configs.load()
    bands = len(split_bands_columns(settings.tile_km))
    group = tile_group.group_name(bands, tile, settings.tile_group_deg)
    return load_group(group, tile.name).get(tile.name, [])


def load_group(group: str, tile: str | None = None) -> dict[str, list[SetCoverage]]:
    """Read one group's measured sets, for every tile or for one alone.

    Args:
        group: The name of the tile group.
        tile: The one tile to read, or None for every tile the group measured.

    Returns:
        coverage: Each tile's sets, observed or not, by tile name, in the order the
            tiles were measured.
    """
    directory = paths.GROUPS_ROOT / group
    filters = None if tile is None else [("tile", "==", tile)]
    measured: dict[str, list[SetCoverage]] = {}
    finished: set[str] = set()
    # A set whose summary never landed was never finished, so it is passed over
    for summary in sorted(directory.glob(f"*{paths.SET_SUMMARY_SUFFIX}")):
        slug = summary.name.removesuffix(paths.SET_SUMMARY_SUFFIX)
        finished.add(slug)
        events = summary.with_name(f"{slug}{paths.EVENTS_SUFFIX}")
        by_tile: dict[str, list[Event]] = {}
        for row in pq.read_table(events, schema=EVENTS, filters=filters).to_pylist():
            by_tile.setdefault(row["tile"], []).append(Event(**row))
        for row in pq.read_table(summary, schema=SUMMARY, filters=filters).to_pylist():
            measured.setdefault(row["tile"], []).append(
                SetCoverage(events=by_tile.get(row["tile"], []), summary=Summary(**row))
            )
    sets = configs.load().instrument_sets
    completed: dict[str, list[SetCoverage]] = {}
    for name, held in measured.items():
        # A configured set that reached none of the tile is shown holding nothing
        blank = replace(
            held[0].summary,
            covered_km2=0.0,
            covered_frac=0.0,
            n_obs=0,
            pixels=0.0,
            t_first=min(one.summary.t_first for one in held),
            t_last=max(one.summary.t_last for one in held),
            span_days=0.0,
        )
        known = {one.summary.set_key for one in held}
        completed[name] = sorted(
            held
            + [
                SetCoverage(
                    events=[],
                    summary=replace(
                        blank,
                        set_key=absent.key,
                        ihid=absent.ihid,
                        iid=absent.iid,
                        pt=absent.pt,
                    ),
                    pending=absent.slug not in finished
                    and file_explorer.has_metadata(group, absent),
                )
                for absent in sets
                if absent.key not in known
            ],
            key=lambda one: (-one.summary.covered_frac, -one.summary.n_obs),
        )
    return completed
