"""The dataset's index: every tile, every stored observation, and the manifest."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from mars_multisensor_dataset.building import paths
from mars_multisensor_dataset.building.metadata import dataset, observation, tile
from mars_multisensor_dataset.building.models.job import Outcome, Plan
from mars_multisensor_dataset.common.disk import parquet


def read_observation_metadata(root: Path) -> list[observation.ObservationMetadata]:
    """Read what every stored observation is, in the order they were written.

    Args:
        root: The directory the index was written in.

    Returns:
        records: One row per tile and observation, none where nothing was written.
    """
    path = root / paths.OBSERVATION_METADATA_NAME
    if not path.exists():
        return []
    return parquet.read_rows(observation.ObservationMetadata, observation.SCHEMA, path)


def write_index(
    plan: Plan,
    collected: Sequence[Outcome],
    root: Path,
    *,
    on_disk: bool,
) -> None:
    """Write the index over every crop of the tiles covered, not this run's alone.

    Args:
        plan: What the build set out to do, whose tiles the index names.
        collected: What every job of this run left.
        root: The dataset's own root directory, made when missing.
        on_disk: Whether an earlier record is kept only while its crop is on disk.
    """
    written = [held for one in collected for held in one.records]
    rewritten = {one.identity for one in written}
    emptied = {name for one in collected for name in one.emptied}
    discarded = {
        (name, one.job.instrument, one.job.identifier)
        for one in collected
        for name in one.discarded
    }
    tiles = {one.identity: one for one in plan.tiles if one.identity not in emptied}
    # What an earlier run left, less what this run rewrote or deleted.
    records = [
        one
        for one in read_observation_metadata(root)
        if one.tile in tiles
        and one.identity not in rewritten
        and one.identity not in discarded
        and (not on_disk or (root / one.path).exists())
    ] + [one for one in written if one.tile in tiles]
    parquet.write_rows(
        list(tiles.values()), tile.SCHEMA, root / paths.TILE_METADATA_NAME
    )
    parquet.write_rows(
        records, observation.SCHEMA, root / paths.OBSERVATION_METADATA_NAME
    )
    # What the dataset holds, which is every instrument in it and not a wish.
    manifest = dataset.dataset_manifest(
        {one.instrument for one in records}, dataset.read_normalization(root)
    )
    dataset.write_manifest(manifest, root)
