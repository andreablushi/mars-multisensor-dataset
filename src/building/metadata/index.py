"""The dataset's index: every tile, every stored observation, and the manifest."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import pyarrow.parquet as pq

from building import paths
from building.metadata import dataset, observation, tile
from building.metadata.observation import ObservationMetadata
from building.models.job import Outcome, Plan
from common.disk import parquet


def read_observation_metadata(root: Path) -> list[ObservationMetadata]:
    """Read what every stored observation is, in the order they were written.

    Args:
        root: The directory the index was written in.

    Returns:
        records: One row per tile and observation, none where nothing was written.
    """
    path = root / paths.OBSERVATION_METADATA_NAME
    if not path.exists():
        return []
    held = pq.read_table(path, schema=observation.SCHEMA)
    return [parquet.build(ObservationMetadata, row) for row in held.to_pylist()]


def write_index(
    plan: Plan,
    collected: Sequence[Outcome],
    root: Path,
    *,
    on_disk: bool,
) -> None:
    """Write the index over every crop of the tiles covered, not this run's alone.

    Args:
        plan: What the build set out to do, whose tiles alone the index names.
        collected: What every job of this run left.
        root: The dataset's own root directory, made when missing.
        on_disk: Whether an earlier record is kept only while its crop is on disk.
    """
    written = [held for one in collected for held in one.records]
    rewritten = {one.identity for one in written}
    tiles = {one.identity: one for one in plan.tiles}
    # What an earlier run left, less what this run rewrote or deleted.
    records = [
        one
        for one in read_observation_metadata(root)
        if one.tile in tiles
        and one.identity not in rewritten
        and (not on_disk or (root / one.path).exists())
    ] + written
    root.mkdir(parents=True, exist_ok=True)
    parquet.write(list(tiles.values()), tile.SCHEMA, root / paths.TILE_METADATA_NAME)
    parquet.write(records, observation.SCHEMA, root / paths.OBSERVATION_METADATA_NAME)
    # What the dataset holds, which is every instrument in it and not a wish.
    manifest = asdict(dataset.dataset_manifest({one.instrument for one in records}))
    (root / paths.DATASET_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
