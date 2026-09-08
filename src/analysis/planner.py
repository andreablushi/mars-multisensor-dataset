"""Turning what a run could do into the jobs it still has to do."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from analysis import paths
from analysis.models.instrument import InstrumentSet
from analysis.models.job import Job, Plan
from analysis.paths import events_path, metadata_file, set_summary_path
from shared.models.feature import Feature


def _outstanding[T, R](
    candidates: Iterable[T],
    output_for: Callable[[T], Path],
    result_for: Callable[[T, Path], R],
    *,
    force: bool,
) -> tuple[tuple[R, ...], int]:
    """Keep every candidate whose output is not already on disk.

    Args:
        candidates: What the run could do, in the order to do it.
        output_for: The file whose presence marks a candidate as finished.
        result_for: Builds what to return for a candidate and its output path.
        force: When True, include candidates that are already finished.

    Returns:
        held: The results for the candidates outstanding.
        skipped: How many were skipped.
    """
    outstanding: list[R] = []
    skipped = 0
    for candidate in candidates:
        output = output_for(candidate)
        if output.exists() and not force:
            skipped += 1
            continue
        outstanding.append(result_for(candidate, output))
    return tuple(outstanding), skipped


def download_plan(
    features: Sequence[Feature],
    instrument_sets: Sequence[InstrumentSet],
    out_root: Path = paths.METADATA_ROOT,
    *,
    force: bool = False,
) -> Plan:
    """Select features and build the download jobs still needed for a run.

    Args:
        features: The full feature catalog.
        instrument_sets: The instrument sets to download for each feature.
        out_root: The metadata output root directory.
        force: When True, include jobs whose output file already exists.

    Returns:
        plan: The plan describing the selection and the jobs to run.
    """
    # Feature selection: a feature the catalogue gives no extent at all is dropped
    usable = [feature for feature in features if not feature.is_point]
    pairs = [
        (feature, instrument_set)
        for feature in usable
        for instrument_set in instrument_sets
    ]
    jobs, skipped = _outstanding(
        pairs,
        lambda pair: metadata_file(out_root, *pair),
        lambda pair, output: Job(
            feature=pair[0], instrument_set=pair[1], output_path=output
        ),
        force=force,
    )
    return Plan(
        jobs=jobs,
        feature_count=len(usable),
        set_count=len(instrument_sets),
        skipped_existing=skipped,
    )


def coverage_plan(
    sources: Sequence[Path],
    features_root: Path = paths.FEATURES_ROOT,
    *,
    force: bool = False,
) -> Plan:
    """Build the coverage jobs still needed for a run.

    Args:
        sources: The instrument set metadata files discovered on disk.
        features_root: The per-feature coverage root directory.
        force: When True, recompute sets that are already done.

    Returns:
        plan: The plan describing the discovery and the jobs to run.
    """
    jobs, skipped = _outstanding(
        sorted(sources, key=lambda path: -path.stat().st_size),
        lambda source: set_summary_path(features_root, source),
        lambda source, output: Job(
            source=source,
            events_path=events_path(features_root, source),
            summary_path=output,
        ),
        force=force,
    )
    return Plan(
        jobs=jobs,
        feature_count=len({source.parent for source in sources}),
        set_count=len(sources),
        skipped_existing=skipped,
    )


def unfinished(
    sources: Sequence[Path], features_root: Path = paths.FEATURES_ROOT
) -> tuple[Path, ...]:
    """Return the instrument sets that still have no coverage artifact.

    Args:
        sources: The instrument set metadata files discovered on disk.
        features_root: The per-feature coverage root directory.

    Returns:
        files: The metadata files with no summary beside them, in discovery order.
    """
    sources_left, _ = _outstanding(
        sources,
        lambda source: set_summary_path(features_root, source),
        lambda source, _output: source,
        force=False,
    )
    return sources_left
