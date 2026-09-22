"""Where the analysis half writes what it measures, and what those files are called."""

from __future__ import annotations

from pathlib import Path

from mars_multisensor_dataset.analysis.models.instrument import InstrumentSet
from mars_multisensor_dataset.common.paths import DATA_ROOT, VERSION_ROOT

ANALYSIS_ROOT = DATA_ROOT / "analysis"
METADATA_ROOT = ANALYSIS_ROOT / "metadata"
COVERAGE_ROOT = ANALYSIS_ROOT / "coverage"
GROUPS_ROOT = COVERAGE_ROOT / "groups"
PANELS_ROOT = ANALYSIS_ROOT / "panels"
SWIM_ROOT = ANALYSIS_ROOT / "swim"

VERSION_ANALYSIS_ROOT = VERSION_ROOT / "analysis"
STATS_ROOT = VERSION_ANALYSIS_ROOT / "stats"
EVALUATION_STATS_ROOT = STATS_ROOT / "evaluation"
SELECTION_ROOT = VERSION_ANALYSIS_ROOT / "selection"
LABELS_ROOT = VERSION_ANALYSIS_ROOT / "labels"

FEATURES_PATH = LABELS_ROOT / "features.jsonl"
VERDICTS_PATH = VERSION_ANALYSIS_ROOT / "verdicts.json"
COVERAGE_INDEX_PATH = COVERAGE_ROOT / "summary.parquet"
DISTORTIONS_PATH = METADATA_ROOT / "summary.parquet"
SELECTED_TILES_PATH = SELECTION_ROOT / "tiles.parquet"
SELECTED_OBSERVATIONS_PATH = SELECTION_ROOT / "observations.parquet"
REFUSED_OBSERVATIONS_PATH = SELECTION_ROOT / "refused.parquet"
SWIM_MAPS_PATH = SWIM_ROOT / "maps.zip"
SWIM_RADAR_PATH = SWIM_ROOT / "radar.zip"

STATS_NAME = "stats.json"
LABELS_NAME = "labels.parquet"
ICE_LABELS_NAME = "ice_labels.parquet"
EVENTS_SUFFIX = ".events.parquet"
SET_SUMMARY_SUFFIX = ".summary.parquet"


def metadata_path(group: str, instrument_set: InstrumentSet) -> Path:
    """Return the JSONL one group's download of one instrument set is stored in."""
    return METADATA_ROOT / group / f"{instrument_set.slug}.jsonl"


def metadata_files() -> list[Path]:
    """Return every non-empty metadata file, one per group and set, sorted."""
    return sorted(
        path for path in METADATA_ROOT.glob("*/*.jsonl") if path.stat().st_size
    )


def coverage_path(source: Path, suffix: str) -> Path:
    """Return the coverage file one metadata file is measured into, by its suffix."""
    return GROUPS_ROOT / source.parent.name / f"{source.stem}{suffix}"
