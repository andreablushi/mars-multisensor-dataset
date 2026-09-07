"""Reading the written metadata back, which a build does to carry it forward."""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

import utils.disk.paths as paths
from building.metadata import feature as features
from building.metadata import observation as records
from building.metadata.feature import FeatureMetadata
from building.metadata.observation import ObservationMetadata
from utils.disk import parquet


def read_feature_metadata(
    root: Path,
) -> dict[tuple[str, str], FeatureMetadata]:
    """Read what the dataset holds about every feature, keyed by the feature.

    Args:
        root: The directory the metadata was written in.

    Returns:
        features: Each feature's own row, by class and name.

    Raises:
        FileNotFoundError: When no features have been written there.
    """
    held = pq.read_table(root / paths.FEATURE_METADATA_NAME, schema=features.SCHEMA)
    return {
        one.identity: one
        for one in (parquet.build(FeatureMetadata, row) for row in held.to_pylist())
    }


def read_observation_metadata(
    root: Path,
) -> list[ObservationMetadata]:
    """Read what every stored observation is, in the order they were written.

    Args:
        root: The directory the metadata was written in.

    Returns:
        records: One row per feature and observation.

    Raises:
        FileNotFoundError: When no observations have been written there.
    """
    held = pq.read_table(root / paths.OBSERVATION_METADATA_NAME, schema=records.SCHEMA)
    return [parquet.build(ObservationMetadata, row) for row in held.to_pylist()]
