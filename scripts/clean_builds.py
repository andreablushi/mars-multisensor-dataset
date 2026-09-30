#!/usr/bin/env python
"""Drop what measures nothing from the published training build, and CRISM's dead bands.

Temporary: run once over the published build, then delete.
"""

from __future__ import annotations

import argparse
import io
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from functools import partial
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
from dhub import archives, submit
from dhub.paths import Artifact
from digitalhub_runtime_python import handler

from analysis.coverage import artifacts as coverage_artifacts
from analysis.selector.merge import merge_track
from analysis.selector.search import best_survey
from analysis.utils.tile_group import tile_grid
from building import paths
from building.configs import crism
from building.dispatcher import INSTRUMENTS
from building.metadata import dataset, index, observation
from building.metadata.observation import ObservationMetadata
from common.config import analysis_settings
from common.disk import parquet

NAME = f"{Artifact.DATASET.published}-training"
BACKUP_SUFFIX = ".before-clean"
LIT_NAME = "crism-lit.jsonl"
PROGRESS_NAME = "crism-bands.jsonl"
ROOT = paths.BUILDING_ROOT / "clean" / NAME
REMOVED_NM = (
    1944.046, 1950.644, 1957.242, 1963.841, 1970.440, 1977.039, 1983.639, 1990.239,
    2003.440, 2010.041, 2023.244, 2043.051, 2069.465, 2076.069, 2082.674, 2089.279,
)  # fmt: skip
KEPT = np.isin(sorted(crism.BANDS_NM + REMOVED_NM), crism.BANDS_NM)
BATCH = 1000
WORKERS = 32
STAGE = SimpleNamespace(
    name="PIPELINE",
    value="scripts.clean_builds:run_cleaning",
    registered="clean-builds",
)


def narrowed_crop(project, key: str) -> str:
    """Put one CRISM crop back on the store without the bands nothing measures.

    Args:
        project: The DigitalHub project the build is published in.
        key: The crop, relative to the build's root.

    Returns:
        key: The crop, once done.
    """
    client, bucket, prefix = archives.stored_folder(project, NAME)
    body = client.get_object(Bucket=bucket, Key=prefix + key)["Body"].read()
    with np.load(io.BytesIO(body)) as held:
        arrays = {name: held[name] for name in held.files}
    if arrays[crism.LAYOUT.measurement].shape[-1] == KEPT.size:
        arrays[crism.LAYOUT.measurement] = arrays[crism.LAYOUT.measurement][..., KEPT]
        arrays["measured_bands"] = arrays["measured_bands"][KEPT]
        written = io.BytesIO()
        np.savez_compressed(written, **arrays)
        client.put_object(Bucket=bucket, Key=prefix + key, Body=written.getvalue())
    return key


def narrowed_record(record: ObservationMetadata) -> ObservationMetadata:
    """Return one CRISM index row on the new band grid, an unmeasured band None.

    Args:
        record: The row as published.

    Returns:
        record: The row with its shape and band columns narrowed.
    """
    counts = [one for one, kept in zip(record.band_valid_count, KEPT) if kept]

    def narrowed(values: tuple) -> tuple:
        kept = [one for one, keep in zip(values, KEPT) if keep]
        return tuple(one if count else None for one, count in zip(kept, counts))

    return replace(
        record,
        shape=(*record.shape[:-1], len(counts)),
        band_mean=narrowed(record.band_mean),
        band_std=narrowed(record.band_std),
        band_valid_count=tuple(counts),
    )


def lost_tiles(records: list[ObservationMetadata]) -> set[str]:
    """Return the tiles the selection no longer keeps without their empty looks.

    Args:
        records: The build's index rows.

    Returns:
        lost: The tiles left without a place.
    """
    looks = {one.identity for one in records if not one.valid_count}
    lit = ROOT / LIT_NAME
    # The CRISM looks the lit conversion dropped measure nothing either
    for row in map(json.loads, lit.read_text().splitlines()):
        if row["lost"] is None:
            band, column, _, stem = row["path"].split("/")
            looks.add((f"{band}_{column}", "CRISM", stem.removesuffix(".npz")))
    window = analysis_settings().window
    grid = tile_grid()
    lost = set()
    for tile in sorted({one.tile for one in records if not one.valid_count}):
        offered = [
            replace(
                one,
                events=[
                    event
                    for event in one.events
                    if (
                        tile,
                        event.iid,
                        INSTRUMENTS[event.iid].observation_id(event.pdsid),
                    )
                    not in looks
                ],
            )
            for one in coverage_artifacts.read_tile_coverage(grid.tile_named(tile))
        ]
        track = merge_track(offered, window)
        if track is None or best_survey(track, window) is None:
            lost.add(tile)
    return lost


@handler()
def run_cleaning(project, workers: int | None = None) -> None:
    """Clean the training build on DigitalHub.

    Args:
        project: The DigitalHub project the build is published in.
        workers: The cores the job was sized with.
    """
    archives.download_artifact(project, Artifact.COVERAGE)
    wanted = [*paths.INDEX_NAMES, LIT_NAME, PROGRESS_NAME]
    archives.download_files(project, NAME, ROOT, wanted)
    records = index.read_observation_metadata(ROOT)
    lost = lost_tiles(records)
    gone = [one for one in records if not one.valid_count or one.tile in lost]
    kept = [one for one in records if one.valid_count and one.tile not in lost]
    print(f"{len(gone):,} rows dropped, {len(lost):,} tiles lost {sorted(lost)}")
    progress = ROOT / PROGRESS_NAME
    done = set(progress.read_text().splitlines()) if progress.exists() else set()
    crops = [
        one.path
        for one in kept
        if one.instrument == crism.LAYOUT.instrument and one.path not in done
    ]
    for start in range(0, len(crops), BATCH):
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            batch = list(
                pool.map(partial(narrowed_crop, project), crops[start : start + BATCH])
            )
        with progress.open("a") as log:
            log.writelines(one + "\n" for one in batch)
        client, bucket, prefix = archives.stored_folder(project, NAME)
        client.upload_file(
            Filename=str(progress), Bucket=bucket, Key=prefix + PROGRESS_NAME
        )
        print(f"{start + len(batch):,} of {len(crops):,} CRISM crops", flush=True)
    kept = [
        narrowed_record(one) if one.instrument == crism.LAYOUT.instrument else one
        for one in kept
    ]
    parquet.write(kept, observation.SCHEMA, ROOT / paths.OBSERVATION_METADATA_NAME)
    tiles = pq.read_table(ROOT / paths.TILE_METADATA_NAME)
    frames = [frame["name"] for frame in tiles["frame"].to_pylist()]
    pq.write_table(
        tiles.filter([one not in lost for one in frames]),
        ROOT / paths.TILE_METADATA_NAME,
        compression="zstd",
    )
    manifest = asdict(dataset.dataset_manifest({one.instrument for one in kept}))
    (ROOT / paths.DATASET_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    client, bucket, prefix = archives.stored_folder(project, NAME)
    for one in paths.INDEX_NAMES:
        key = prefix + one
        client.copy_object(
            Bucket=bucket,
            Key=key + BACKUP_SUFFIX,
            CopySource={"Bucket": bucket, "Key": key},
        )
        client.upload_file(Filename=str(ROOT / one), Bucket=bucket, Key=key)
    # Only once the index names them no more, so a reader never misses one
    for one in gone:
        client.delete_object(Bucket=bucket, Key=prefix + one.path)
    print(f"index rewritten, {len(gone):,} crops deleted, done", flush=True)


def main() -> int:
    """Submit the cleaning to DigitalHub.

    Returns:
        code: A process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default="clean", help="branch the job clones")
    return submit.submitted(STAGE, parser.parse_args().ref)


if __name__ == "__main__":
    raise SystemExit(main())
