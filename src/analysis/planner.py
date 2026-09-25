"""The download and coverage jobs a run still has to do."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from analysis.models.instrument import InstrumentSet
from analysis.models.job import CoverageJob, DownloadJob, Plan
from analysis.models.tile_group import TileGroup
from analysis.paths import (
    EVENTS_SUFFIX,
    SET_SUMMARY_SUFFIX,
    coverage_path,
    metadata_path,
)


def download_plan(
    groups: Sequence[TileGroup],
    instrument_sets: Sequence[InstrumentSet],
    *,
    force: bool = False,
) -> Plan:
    """Build the download jobs still needed for a run.

    Args:
        groups: Every group the tiles are grouped into.
        instrument_sets: The instrument sets to download for each group.
        force: When True, include jobs whose output file already exists.

    Returns:
        plan: The jobs to run, and how many were already downloaded.
    """
    jobs = [
        DownloadJob(
            group=group,
            instrument_set=instrument_set,
            output_path=metadata_path(group.name, instrument_set),
        )
        for group in groups
        for instrument_set in instrument_sets
    ]
    return Plan.of(jobs, [force or not job.output_path.exists() for job in jobs])


def coverage_plan(
    sources: Sequence[Path], groups: Sequence[TileGroup], *, force: bool = False
) -> Plan:
    """Build the coverage jobs still needed for a run, the largest set first.

    Args:
        sources: The instrument set metadata files discovered on disk.
        groups: Every group, which each source is matched to by its directory.
        force: When True, recompute sets that are already done.

    Returns:
        plan: The jobs to run, and how many were already measured.
    """
    named = {group.name: group for group in groups}
    jobs = [
        CoverageJob(
            group=named[source.parent.name],
            source=source,
            events_path=coverage_path(source, EVENTS_SUFFIX),
            summary_path=coverage_path(source, SET_SUMMARY_SUFFIX),
        )
        for source in sorted(sources, key=lambda path: -path.stat().st_size)
    ]
    return Plan.of(jobs, [force or not job.summary_path.exists() for job in jobs])


def unmeasured_sources(sources: Sequence[Path]) -> list[Path]:
    """Return the metadata files with no coverage summary written for them yet."""
    return [
        source
        for source in sources
        if not coverage_path(source, SET_SUMMARY_SUFFIX).exists()
    ]
