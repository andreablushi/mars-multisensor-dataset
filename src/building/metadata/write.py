"""Writing the metadata down: what the dataset is, and what was taken of it."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from building import paths
from building.metadata import dataset
from building.metadata import observation as records
from building.metadata import tile as tiles
from building.metadata.observation import ObservationMetadata
from building.metadata.tile import TileMetadata
from common.disk import parquet


def write_metadata(
    held: Sequence[TileMetadata],
    taken: Sequence[ObservationMetadata],
    instruments: tuple[str, ...],
    band_centres_nm: dict[str, tuple[float, ...]],
    root: Path,
) -> None:
    """Write down what the dataset is, every tile in it, and every observation.

    Args:
        held: One row per tile, in the order to write them.
        taken: One record per tile and observation, in the same manner.
        instruments: The instruments the build covered.
        band_centres_nm: The band grid of each instrument with a wavelength.
        root: The directory the files are written in, made when missing.
    """
    root.mkdir(parents=True, exist_ok=True)
    parquet.write(held, tiles.SCHEMA, root / paths.TILE_METADATA_NAME)
    parquet.write(taken, records.SCHEMA, root / paths.OBSERVATION_METADATA_NAME)
    manifest = asdict(dataset.dataset_manifest(instruments, band_centres_nm))
    (root / paths.DATASET_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
