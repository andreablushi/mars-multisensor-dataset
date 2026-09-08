"""Writing the metadata down: what the dataset is, and what was taken of it."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from building import paths
from building.metadata import dataset
from building.metadata import feature as features
from building.metadata import observation as records
from building.metadata.feature import FeatureMetadata
from building.metadata.observation import ObservationMetadata
from shared.disk import parquet


def write_metadata(
    held: Sequence[FeatureMetadata],
    taken: Sequence[ObservationMetadata],
    instruments: tuple[str, ...],
    root: Path,
) -> None:
    """Write down what the dataset is, every feature in it, and every observation.

    Args:
        held: One row per feature, in the order to write them.
        taken: One record per feature and observation, in the same manner.
        instruments: The instruments the build covered.
        root: The directory the files are written in, made when missing.
    """
    root.mkdir(parents=True, exist_ok=True)
    parquet.write(held, features.SCHEMA, root / paths.FEATURE_METADATA_NAME)
    parquet.write(taken, records.SCHEMA, root / paths.OBSERVATION_METADATA_NAME)
    manifest = asdict(dataset.dataset_manifest(instruments))
    (root / paths.DATASET_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
