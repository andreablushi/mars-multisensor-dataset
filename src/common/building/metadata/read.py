"""Reading the written metadata back, which a build does to carry it forward."""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from common.building import paths
from common.building.metadata import observation as records
from common.building.metadata.observation import ObservationMetadata
from common.disk import parquet


def read_observation_metadata(
    root: Path,
) -> list[ObservationMetadata]:
    """Read what every stored observation is, in the order they were written.

    Args:
        root: The directory the metadata was written in.

    Returns:
        records: One row per tile and observation.

    Raises:
        FileNotFoundError: When no observations have been written there.
    """
    held = pq.read_table(root / paths.OBSERVATION_METADATA_NAME, schema=records.SCHEMA)
    return [parquet.build(ObservationMetadata, row) for row in held.to_pylist()]
