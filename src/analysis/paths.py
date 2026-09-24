"""Where the analysis half writes what it measures, and what those files are called."""

from __future__ import annotations

from pathlib import Path

from analysis.models.instrument import InstrumentSet
from common.paths import DATA_ROOT

ANALYSIS_ROOT = DATA_ROOT / "analysis"
METADATA_ROOT = ANALYSIS_ROOT / "metadata"
COVERAGE_ROOT = ANALYSIS_ROOT / "coverage"
GROUPS_ROOT = COVERAGE_ROOT / "groups"
STATS_ROOT = ANALYSIS_ROOT / "stats"
EVALUATION_STATS_ROOT = STATS_ROOT / "evaluation"
SELECTION_ROOT = ANALYSIS_ROOT / "selection"
LABELS_ROOT = ANALYSIS_ROOT / "labels"

FEATURES_PATH = LABELS_ROOT / "features.jsonl"
VERDICTS_PATH = ANALYSIS_ROOT / "verdicts.json"
COVERAGE_SUMMARY_PATH = COVERAGE_ROOT / "summary.parquet"
DISTORTIONS_PATH = METADATA_ROOT / "summary.parquet"
SELECTED_TILES_PATH = SELECTION_ROOT / "tiles.parquet"
SELECTED_OBSERVATIONS_PATH = SELECTION_ROOT / "observations.parquet"

STATS_NAME = "stats.json"
LABELS_NAME = "labels.parquet"
EVENTS_SUFFIX = ".events.parquet"
SET_SUMMARY_SUFFIX = ".summary.parquet"


def metadata_path(group: str, instrument_set: InstrumentSet) -> Path:
    """Return the JSONL one group's download of one instrument set is stored in."""
    return METADATA_ROOT / group / f"{instrument_set.slug}.jsonl"


def coverage_path(source: Path, suffix: str) -> Path:
    """Return the coverage file one metadata file is measured into, by its suffix."""
    return GROUPS_ROOT / source.parent.name / f"{source.stem}{suffix}"
