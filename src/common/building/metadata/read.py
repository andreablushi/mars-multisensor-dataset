"""Reading the written metadata back, which a build does to carry it forward."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pyarrow.parquet as pq

from common.building import paths
from common.building.metadata import observation as records
from common.building.metadata import tile as tiles
from common.building.metadata.observation import ObservationMetadata
from common.building.metadata.tile import TileMetadata
from common.disk import parquet


def read_tile_metadata(
    root: Path,
) -> dict[str, TileMetadata]:
    """Read what the dataset holds about every tile, keyed by the tile.

    Args:
        root: The directory the metadata was written in.

    Returns:
        tiles: Each tile's own row, by name, the centre of each
            taken from its own box rather than from a file that may predate it.

    Raises:
        FileNotFoundError: When no tiles have been written there.
    """
    held = pq.read_table(root / paths.TILE_METADATA_NAME, schema=tiles.SCHEMA)
    return {
        one.identity: replace(
            one, centre_lon=one.frame.centre_lon, centre_lat=one.frame.centre_lat
        )
        for one in (parquet.build(TileMetadata, row) for row in held.to_pylist())
    }


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
