"""The coverage artifacts: each set's files, the index gathering them, and reads."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq

from analysis import paths
from analysis.coverage.models.coverage import Event, SetCoverage
from analysis.coverage.models.summary import Summary
from analysis.models.job import CoverageJob
from analysis.utils.tile_group import group_of
from common.config import analysis_settings
from common.disk import parquet
from common.disk.files import atomic_path
from common.models.tile import Tile

EVENTS = parquet.schema_of(Event)
SUMMARY = parquet.schema_of(Summary)


def write_coverage(
    job: CoverageJob, events: Sequence[Event], summaries: Sequence[Summary]
) -> None:
    """Write one set's observation rows and the row describing it on each tile.

    Args:
        job: The instrument set that was computed, naming both destinations.
        events: The set's observation rows, tile by tile in chronological order.
        summaries: One row per tile the set reached, describing it as a whole.
    """
    parquet.write(events, EVENTS, job.events_path)
    parquet.write(summaries, SUMMARY, job.summary_path)


def reindex() -> None:
    """Rebuild the grid-wide summary from every group's summaries on disk."""
    summary_paths = sorted(paths.GROUPS_ROOT.glob(f"*/*{paths.SET_SUMMARY_SUFFIX}"))
    tables = [pq.read_table(path, schema=SUMMARY) for path in summary_paths]
    combined = pa.concat_tables(tables) if tables else SUMMARY.empty_table()
    with atomic_path(paths.COVERAGE_SUMMARY_PATH) as tmp:
        pq.write_table(combined, tmp, compression="zstd")


def catalogued_rows() -> list[Summary]:
    """Read every row of the grid-wide summary.

    Returns:
        rows: One row per tile and instrument set measured, in index order.
    """
    path = paths.COVERAGE_SUMMARY_PATH
    if not path.exists():
        return []
    return [Summary(**row) for row in pq.read_table(path, schema=SUMMARY).to_pylist()]


def measured_groups() -> list[str]:
    """Name every tile group holding a measured set on disk, busiest first."""
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
    """Read every instrument set for one tile, widest then busiest first."""
    return load_group(group_of(tile), tile.name).get(tile.name, [])


def load_group(group: str, tile: str | None = None) -> dict[str, list[SetCoverage]]:
    """Read one group's measured sets, for every tile or for one alone.

    Args:
        group: The name of the tile group.
        tile: The one tile to read, or None for every tile the group measured.

    Returns:
        coverage: Each tile's sets by tile name, widest then busiest first, with a
            blank for every configured set that reached none of it.
    """
    directory = paths.GROUPS_ROOT / group
    filters = None if tile is None else [("tile", "==", tile)]
    measured: dict[str, list[SetCoverage]] = {}
    # A set whose summary never landed was never finished, so it is passed over
    for summary_path in sorted(directory.glob(f"*{paths.SET_SUMMARY_SUFFIX}")):
        slug = summary_path.name.removesuffix(paths.SET_SUMMARY_SUFFIX)
        events_path = summary_path.with_name(f"{slug}{paths.EVENTS_SUFFIX}")
        events = pq.read_table(events_path, schema=EVENTS, filters=filters)
        summaries = pq.read_table(summary_path, schema=SUMMARY, filters=filters)
        events_by_tile: dict[str, list[Event]] = {}
        for row in events.to_pylist():
            events_by_tile.setdefault(row["tile"], []).append(Event(**row))
        for row in summaries.to_pylist():
            measured.setdefault(row["tile"], []).append(
                SetCoverage(
                    events=events_by_tile.get(row["tile"], []), summary=Summary(**row)
                )
            )
    configured = analysis_settings().instrument_sets
    completed: dict[str, list[SetCoverage]] = {}
    for tile_name, reached in measured.items():
        # A configured set that reached none of the tile is shown holding nothing
        blank = replace(
            reached[0].summary,
            covered_frac=0.0,
            n_obs=0,
            t_first=min(instrument.summary.t_first for instrument in reached),
            t_last=max(instrument.summary.t_last for instrument in reached),
        )
        reached_keys = {instrument.summary.set_key for instrument in reached}
        blanks = [
            SetCoverage(
                events=[], summary=replace(blank, set_key=absent.key, iid=absent.iid)
            )
            for absent in configured
            if absent.key not in reached_keys
        ]
        completed[tile_name] = sorted(
            reached + blanks,
            key=lambda instrument: (
                -instrument.summary.covered_frac,
                -instrument.summary.n_obs,
            ),
        )
    return completed
