"""Where the analysis half writes what it measures, and what those files are called."""

from __future__ import annotations

from pathlib import Path

from analysis.models.instrument import InstrumentSet
from common.paths import CONFIGS_ROOT, DATA_ROOT

CONFIG_PATH = CONFIGS_ROOT / "analysis.yaml"

ANALYSIS_ROOT = DATA_ROOT / "analysis"
METADATA_ROOT = ANALYSIS_ROOT / "metadata"
COVERAGE_ROOT = ANALYSIS_ROOT / "coverage"
GROUPS_ROOT = COVERAGE_ROOT / "groups"
STATS_ROOT = ANALYSIS_ROOT / "stats"
EVALUATION_STATS_ROOT = STATS_ROOT / "evaluation"
SELECTION_ROOT = ANALYSIS_ROOT / "selection"
LABELS_ROOT = ANALYSIS_ROOT / "labels"
FEATURES_PATH = LABELS_ROOT / "features.jsonl"
SOUNDINGS_PATH = METADATA_ROOT / "soundings.jsonl"

STATS_NAME = "stats.json"
SELECTED_TILES_NAME = "tiles.parquet"
SELECTED_OBSERVATIONS_NAME = "observations.parquet"
SUMMARY_NAME = "summary.parquet"
LABELS_NAME = "labels.parquet"
EVENTS_SUFFIX = ".events.parquet"
SET_SUMMARY_SUFFIX = ".summary.parquet"


def metadata_file(root: Path, group: str, instrument_set: InstrumentSet) -> Path:
    """Return the JSONL path for one group and instrument set.

    Args:
        root: The metadata root directory.
        group: The name of the group being stored.
        instrument_set: The instrument set being stored.

    Returns:
        path: The path to the JSONL output file.
    """
    return root / group / f"{instrument_set.slug}.jsonl"


def events_path(root: Path, source: Path) -> Path:
    """Return the per-observation events file for one instrument set.

    Args:
        root: The per-group coverage root directory.
        source: The instrument set's metadata JSONL file.

    Returns:
        path: The path to the events parquet file.
    """
    return root / source.parent.name / f"{source.stem}{EVENTS_SUFFIX}"


def set_summary_path(root: Path, source: Path) -> Path:
    """Return the summary file for one instrument set.

    Args:
        root: The per-group coverage root directory.
        source: The instrument set's metadata JSONL file.

    Returns:
        path: The path to the summary parquet file.
    """
    return root / source.parent.name / f"{source.stem}{SET_SUMMARY_SUFFIX}"


def catalog_summary_path(root: Path = COVERAGE_ROOT) -> Path:
    """Return the file holding every tile's summary rows together.

    Args:
        root: The coverage root directory.

    Returns:
        path: The path to the grid-wide summary parquet file.
    """
    return root / SUMMARY_NAME
