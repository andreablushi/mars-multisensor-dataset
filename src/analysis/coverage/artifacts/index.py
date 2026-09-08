"""The coverage artifacts a run left: gathering them into an index, and reading it."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

import shared.disk.paths as paths
from analysis.coverage.artifacts.write import EVENTS, SUMMARY
from analysis.coverage.models.coverage import Event, SetCoverage
from analysis.coverage.models.summary import Summary
from analysis.metadata import file_explorer
from analysis.models.instrument import InstrumentSet
from shared.disk.files import atomic_path


def reindex() -> int:
    """Rebuild the per-feature and catalogue-wide summaries from disk.

    Returns:
        rows: How many summary rows the catalogue index holds.
    """
    for feature_dir in sorted(paths.FEATURES_ROOT.glob("*/*")):
        found = sorted(feature_dir.glob(f"*{paths.SET_SUMMARY_SUFFIX}"))
        if found:
            _concatenate(found, feature_dir / paths.SUMMARY_NAME)
    return _concatenate(
        sorted(paths.FEATURES_ROOT.glob(f"*/*/{paths.SUMMARY_NAME}")),
        paths.catalog_summary_path(),
    )


def computed_features() -> set[tuple[str, str]]:
    """Return every feature that has coverage computed locally.

    Returns:
        features: The class and name slug of each feature with a computed set.
    """
    return {
        (path.parent.parent.name, path.parent.name)
        for path in paths.FEATURES_ROOT.glob(f"*/*/*{paths.EVENTS_SUFFIX}")
    }


def catalogued_features() -> list[tuple[str, str]]:
    """Name every feature the computed artifacts hold, as the catalogue spells it.

    Returns:
        features: The class and name of each feature, once each and in order.
    """
    table = _index()
    return sorted(
        dict.fromkeys(
            zip(
                table.column("feature_class").to_pylist(),
                table.column("feature_name").to_pylist(),
                strict=True,
            )
        )
    )


def catalogued_observations() -> dict[tuple[str, str], int]:
    """Count the observations the computed artifacts hold for each feature.

    Returns:
        observations: How many observations each feature holds, by class and name.
    """
    table = _index()
    counted: dict[tuple[str, str], int] = {}
    for feature_class, name, observations in zip(
        table.column("feature_class").to_pylist(),
        table.column("feature_name").to_pylist(),
        table.column("n_obs").to_pylist(),
        strict=True,
    ):
        counted[(feature_class, name)] = counted.get((feature_class, name), 0) + (
            observations or 0
        )
    return counted


def catalogued_rows() -> list[Summary]:
    """Read every row the computed artifacts hold anywhere.

    Returns:
        rows: One row per feature and instrument set measured, in index order.
    """
    return [Summary(**row) for row in _index().to_pylist()]


def load_feature(feature_class: str, name: str) -> list[SetCoverage]:
    """Read every instrument set for one feature, observed or not.

    Args:
        feature_class: The feature class, such as Crater.
        name: The feature name as ODE spells it.

    Returns:
        coverage: One entry per instrument set, widest coverage first, then busiest.
    """
    directory = paths.feature_coverage_dir(paths.FEATURES_ROOT, feature_class, name)
    measured: list[SetCoverage] = []
    for events in sorted(directory.glob(f"*{paths.EVENTS_SUFFIX}")):
        # A set whose summary never landed was never finished, so it is passed over
        summary = events.with_name(
            events.name.replace(paths.EVENTS_SUFFIX, paths.SET_SUMMARY_SUFFIX)
        )
        if not summary.exists():
            continue
        measured.append(
            SetCoverage(
                events=[
                    Event(**row)
                    for row in pq.read_table(events, schema=EVENTS).to_pylist()
                ],
                summary=Summary(
                    **pq.read_table(summary, schema=SUMMARY).to_pylist()[0]
                ),
            )
        )
    if not measured:
        return []
    # A set the artifacts hold anywhere but not here is shown holding nothing
    blank = replace(
        measured[0].summary,
        covered_km2=0.0,
        covered_frac=0.0,
        n_obs=0,
        pixels=0.0,
        t_first=min(one.summary.t_first for one in measured),
        t_last=max(one.summary.t_last for one in measured),
        span_days=0.0,
    )
    known = {one.summary.set_key for one in measured}
    return sorted(
        measured
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
                pending=file_explorer.has_metadata(directory, absent),
            )
            for absent in (
                InstrumentSet.from_key(key)
                for key in dict.fromkeys(_index().column("set_key").to_pylist())
                if key not in known
            )
        ],
        key=lambda one: (-one.summary.covered_frac, -one.summary.n_obs),
    )


def _concatenate(found: list[Path], destination: Path) -> int:
    """Write many summary files out as one, atomically.

    Args:
        found: The summary parquet files to combine, in the order to keep.
        destination: The parquet file to write them to.

    Returns:
        rows: The number of rows written.
    """
    tables = [pq.read_table(path, schema=SUMMARY) for path in found]
    combined = pa.concat_tables(tables) if tables else SUMMARY.empty_table()
    with atomic_path(destination) as tmp:
        pq.write_table(combined, tmp, compression="zstd")
    return combined.num_rows


def _index() -> pa.Table:
    """Read the catalogue index, under the current schema.

    Returns:
        index: The index, empty when no run has left one.
    """
    path = paths.catalog_summary_path()
    return (
        pq.read_table(path, schema=SUMMARY) if path.exists() else SUMMARY.empty_table()
    )
