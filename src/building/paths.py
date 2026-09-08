"""Where a build writes what it makes, and what those files are called."""

from __future__ import annotations

from pathlib import Path

from shared.disk.slugify import slugify
from shared.paths import CONFIGS_ROOT, DATA_ROOT

CONFIG_PATH = CONFIGS_ROOT / "building.yaml"

BUILDING_ROOT = DATA_ROOT / "building"
DATASETS_ROOT = BUILDING_ROOT / "dataset"

PREPROCESSING_ROOT = BUILDING_ROOT / "preprocessing"
CRISM_ROOT = PREPROCESSING_ROOT / "crism"
SHARAD_ROOT = PREPROCESSING_ROOT / "sharad"
MOLA_ROOT = PREPROCESSING_ROOT / "mola"
CTX_ROOT = PREPROCESSING_ROOT / "ctx"

FEATURE_METADATA_NAME = "features.parquet"
OBSERVATION_METADATA_NAME = "observations.parquet"
DATASET_MANIFEST_NAME = "dataset.json"
SAMPLE_SUFFIX = ".npz"


def dataset_root(name: str, root: Path = DATASETS_ROOT) -> Path:
    """Return where one named build of the dataset is written.

    Args:
        name: What the build is called, as its config names it.
        root: The directory every build of the dataset is written under.

    Returns:
        path: The directory that build owns, which need not exist.
    """
    return root / slugify(name)
