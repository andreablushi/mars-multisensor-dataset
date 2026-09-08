"""Where the analysis half writes what it measures, and what those files are called."""

from __future__ import annotations

from pathlib import Path

from analysis.models.feature import Feature
from analysis.models.instrument import InstrumentSet
from shared.disk.slugify import slugify
from shared.paths import CONFIGS_ROOT, DATA_ROOT

CONFIG_PATH = CONFIGS_ROOT / "analysis.yaml"

CATALOG_ROOT = DATA_ROOT / "_catalog"

ANALYSIS_ROOT = DATA_ROOT / "analysis"
METADATA_ROOT = ANALYSIS_ROOT / "metadata"
COVERAGE_ROOT = ANALYSIS_ROOT / "coverage"
FEATURES_ROOT = COVERAGE_ROOT / "features"
STATS_ROOT = ANALYSIS_ROOT / "stats"
SELECTION_ROOT = ANALYSIS_ROOT / "selection"

STATS_NAME = "stats.json"
SELECTED_FEATURES_NAME = "features.parquet"
SELECTED_OBSERVATIONS_NAME = "observations.parquet"
FEATURES_CACHE_NAME = "features.jsonl"
SUMMARY_NAME = "summary.parquet"
EVENTS_SUFFIX = ".events.parquet"
SET_SUMMARY_SUFFIX = ".summary.parquet"


def metadata_file(root: Path, feature: Feature, instrument_set: InstrumentSet) -> Path:
    """Return the JSONL path for one feature and instrument set.

    Args:
        root: The metadata root directory.
        feature: The feature being stored.
        instrument_set: The instrument set being stored.

    Returns:
        path: The path to the JSONL output file.
    """
    directory = root / slugify(feature.feature_class) / slugify(feature.name)
    return directory / f"{instrument_set.slug}.jsonl"


def feature_coverage_dir(root: Path, feature_class: str, name: str) -> Path:
    """Return where one feature's artifacts live, from its catalogue names.

    Args:
        root: The artifacts subtree the path is built under.
        feature_class: The feature class, such as Crater.
        name: The feature name as ODE spells it.

    Returns:
        path: The feature's directory under that root, which need not exist.
    """
    return root / slugify(feature_class) / slugify(name)


def events_path(root: Path, source: Path) -> Path:
    """Return the per-observation events file for one instrument set.

    Args:
        root: The per-feature coverage root directory.
        source: The instrument set's metadata JSONL file.

    Returns:
        path: The path to the events parquet file.
    """
    held = source.parent
    return root / held.parent.name / held.name / f"{source.stem}{EVENTS_SUFFIX}"


def set_summary_path(root: Path, source: Path) -> Path:
    """Return the summary file for one instrument set.

    Args:
        root: The per-feature coverage root directory.
        source: The instrument set's metadata JSONL file.

    Returns:
        path: The path to the summary parquet file.
    """
    held = source.parent
    return root / held.parent.name / held.name / f"{source.stem}{SET_SUMMARY_SUFFIX}"


def catalog_summary_path(root: Path = COVERAGE_ROOT) -> Path:
    """Return the file holding every feature's summary rows together.

    Args:
        root: The coverage root directory.

    Returns:
        path: The path to the catalogue-wide summary parquet file.
    """
    return root / SUMMARY_NAME


def features_path(cache_dir: Path = CATALOG_ROOT) -> Path:
    """Return where the cached feature catalogue lives.

    Args:
        cache_dir: Directory holding the cached catalogue files.

    Returns:
        path: The path to the features JSONL file, which need not exist.
    """
    return cache_dir / FEATURES_CACHE_NAME
