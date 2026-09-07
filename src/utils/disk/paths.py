"""Where the project lives on disk, what its files are called, and where they go."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from utils.disk.slugify import slugify

if TYPE_CHECKING:
    from analysis.models.feature import Feature
    from analysis.models.instrument import InstrumentSet

REPO_ROOT = Path(__file__).resolve().parents[3]

CONFIGS_ROOT = REPO_ROOT / "configs"
RUNNER_CONFIG_PATH = CONFIGS_ROOT / "analysis_runner.yaml"
BUILDING_CONFIG_PATH = CONFIGS_ROOT / "building_runner.yaml"
FILTER_CONFIG_PATH = CONFIGS_ROOT / "window_filter.yaml"
PLATFORM_CONFIG_PATH = CONFIGS_ROOT / "digitalhub.yaml"

DATA_ROOT = REPO_ROOT / "data"
CATALOG_ROOT = DATA_ROOT / "_catalog"

ANALYSIS_ROOT = DATA_ROOT / "analysis"
METADATA_ROOT = ANALYSIS_ROOT / "metadata"
COVERAGE_ROOT = ANALYSIS_ROOT / "coverage"
FEATURES_ROOT = COVERAGE_ROOT / "features"
STATS_ROOT = ANALYSIS_ROOT / "stats"
SELECTION_ROOT = ANALYSIS_ROOT / "selection"

BUILDING_ROOT = DATA_ROOT / "building"
DATASETS_ROOT = BUILDING_ROOT / "dataset"
PREPROCESSING_ROOT = BUILDING_ROOT / "preprocessing"
CRISM_ROOT = PREPROCESSING_ROOT / "crism"
SHARAD_ROOT = PREPROCESSING_ROOT / "sharad"
MOLA_ROOT = PREPROCESSING_ROOT / "mola"
CTX_ROOT = PREPROCESSING_ROOT / "ctx"

STATS_NAME = "stats.json"
FEATURE_METADATA_NAME = "features.parquet"
OBSERVATION_METADATA_NAME = "observations.parquet"
DATASET_MANIFEST_NAME = "dataset.json"
SAMPLE_SUFFIX = ".npz"
SELECTED_FEATURES_NAME = "features.parquet"
SELECTED_OBSERVATIONS_NAME = "observations.parquet"
FEATURES_CACHE_NAME = "features.jsonl"
SUMMARY_NAME = "summary.parquet"
EVENTS_SUFFIX = ".events.parquet"
SET_SUMMARY_SUFFIX = ".summary.parquet"


def dataset_root(name: str, root: Path = DATASETS_ROOT) -> Path:
    """Return where one named build of the dataset is written.

    Args:
        name: What the build is called, as its config names it.
        root: The directory every build of the dataset is written under.

    Returns:
        path: The directory that build owns, which need not exist.
    """
    return root / slugify(name)


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


def _mirrored(root: Path, feature_dir: Path) -> Path:
    """Return the directory mirroring one feature's metadata under a root.

    Args:
        root: The artifacts subtree the path is built under.
        feature_dir: The feature's metadata directory.

    Returns:
        path: The matching path under the given root.
    """
    return root / feature_dir.parent.name / feature_dir.name


def events_path(root: Path, source: Path) -> Path:
    """Return the per-observation events file for one instrument set.

    Args:
        root: The per-feature coverage root directory.
        source: The instrument set's metadata JSONL file.

    Returns:
        path: The path to the events parquet file.
    """
    return _mirrored(root, source.parent) / f"{source.stem}{EVENTS_SUFFIX}"


def set_summary_path(root: Path, source: Path) -> Path:
    """Return the summary file for one instrument set.

    Args:
        root: The per-feature coverage root directory.
        source: The instrument set's metadata JSONL file.

    Returns:
        path: The path to the summary parquet file.
    """
    return _mirrored(root, source.parent) / f"{source.stem}{SET_SUMMARY_SUFFIX}"


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
