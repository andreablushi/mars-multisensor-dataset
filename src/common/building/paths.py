"""Where a build writes what it makes, and what those files are called."""

from __future__ import annotations

from pathlib import Path

from common.disk.slugify import slugify
from common.paths import COMMON_CONFIGS_ROOT, DATA_ROOT

CONFIG_PATH = COMMON_CONFIGS_ROOT / "building.yaml"

BUILDING_ROOT = DATA_ROOT / "building"
DATASETS_ROOT = BUILDING_ROOT / "dataset"

PREPROCESSING_ROOT = BUILDING_ROOT / "preprocessing"
CRISM_ROOT = PREPROCESSING_ROOT / "crism"
SHARAD_ROOT = PREPROCESSING_ROOT / "sharad"
MOLA_ROOT = PREPROCESSING_ROOT / "mola"
CTX_ROOT = PREPROCESSING_ROOT / "ctx"

TILE_METADATA_NAME = "tiles.parquet"
OBSERVATION_METADATA_NAME = "observations.parquet"
DATASET_MANIFEST_NAME = "dataset.json"
SAMPLE_SUFFIX = ".npz"
INDEX_NAMES = (TILE_METADATA_NAME, OBSERVATION_METADATA_NAME, DATASET_MANIFEST_NAME)


def dataset_root(name: str, root: Path = DATASETS_ROOT) -> Path:
    """Return where one named build of the dataset is written.

    Args:
        name: What the build is called, as its config names it.
        root: The directory every build of the dataset is written under.

    Returns:
        path: The directory that build owns, which need not exist.
    """
    return root / slugify(name)


def crop_paths(root: Path) -> list[Path]:
    """Return every crop one build of the dataset holds on disk.

    Args:
        root: The dataset's own root directory.

    Returns:
        crops: The file of each crop written there, in no particular order.
    """
    return list(root.rglob(f"*{SAMPLE_SUFFIX}"))
