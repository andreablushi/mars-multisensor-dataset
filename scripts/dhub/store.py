"""The folders a platform run publishes to the store, and reads back from it."""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from pathlib import Path
from urllib.parse import urlparse

import digitalhub as dh
from digitalhub.stores.data.api import get_default_store

from mars_multisensor_dataset.analysis import paths as analysis_paths
from mars_multisensor_dataset.building import paths as building_paths
from mars_multisensor_dataset.common.config import training_settings
from mars_multisensor_dataset.common.paths import VERSION
from mars_multisensor_dataset.common.pool import cancellable_pool

ARTIFACTS = {
    "coverage": analysis_paths.COVERAGE_ROOT,
    "metadata": analysis_paths.METADATA_ROOT,
    "selection": analysis_paths.SELECTION_ROOT,
    "stats": analysis_paths.STATS_ROOT,
    "labels": analysis_paths.LABELS_ROOT,
    "swim": analysis_paths.SWIM_ROOT,
}
SHARED = ("coverage", "metadata", "swim")
VERDICTS = (analysis_paths.VERDICTS_PATH, analysis_paths.LAVAFLOW_VERDICTS_PATH)


def published_name(name: str) -> str:
    """Return what one folder is published as, prefixed by its version unless shared."""
    return name if name in SHARED else f"{VERSION}-{name}"


def folder_key(name: str) -> str:
    """Return the folder one artifact is kept in, under its version unless shared."""
    return name if name in SHARED else f"{VERSION}/{name}"


def stored_folder(project, name: str) -> tuple[object, str, str]:
    """Return the S3 client, bucket and key prefix one published folder lives under.

    Args:
        project: The DigitalHub project the folder belongs to.
        name: The folder, its version added unless shared.

    Returns:
        client: The S3 client, its credentials fresh.
        bucket: The bucket the folder is kept in.
        prefix: The key every object of the folder starts with.
    """
    # A bare S3 client never refreshes itself, unlike the platform's own calls
    dh.refresh_token()
    root = get_default_store(project.name)
    place = urlparse(f"{root}/{project.name}/artifacts/{folder_key(name)}/")
    return dh.get_s3_client(), place.netloc, place.path.lstrip("/")


def upload_folder(
    project,
    name: str,
    root: Path,
    files: Sequence[Path] | None = None,
    last: Sequence[Path] = (),
    uploads: int = 16,
):
    """Publish files of one directory as a folder, many at once and a few after them.

    Args:
        project: The DigitalHub project to log the folder into.
        name: The folder, its version added unless shared.
        root: The directory the files sit in, which keys are taken relative to.
        files: What to send in any order, or None for every file under the root.
        last: What is sent one by one, only once every other file is up.
        uploads: How many files are sent at once.

    Returns:
        artifact: The logged artifact.
    """
    client, bucket, prefix = stored_folder(project, name)
    files = (
        [one for one in root.rglob("*") if one.is_file()] if files is None else files
    )
    going = sum(path.stat().st_size for path in [*files, *last])
    told = f"{len(files) + len(last):,} files, {going / 1e6:.0f} MB"
    print(f"uploading {name}, {told}", flush=True)
    keys = [prefix + path.relative_to(root).as_posix() for path in [*files, *last]]
    with cancellable_pool(ThreadPoolExecutor(uploads)) as sending:
        list(sending.map(client.upload_file, map(str, files), repeat(bucket), keys))
    for path, key in zip(last, keys[len(files) :], strict=True):
        client.upload_file(str(path), bucket, key)
    return project.new_artifact(
        name=published_name(name), kind="artifact", path=f"s3://{bucket}/{prefix}"
    )


def download_folder(
    project,
    name: str,
    into: Path,
    names: Sequence[str] | None = None,
    downloads: int = 16,
) -> None:
    """Bring a published folder back to disk, whole or only some files at its top.

    Args:
        project: The DigitalHub project the folder was logged into.
        name: The folder, its version added unless shared, possibly not yet published.
        into: The directory it lands in, replaced when the whole folder comes down.
        names: The files to bring down from the top of the folder, those missing
            skipped, or None for the whole folder.
        downloads: How many files are brought down at once.
    """
    client, bucket, prefix = stored_folder(project, name)
    listed = client.get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix=prefix, **({"Delimiter": "/"} if names else {})
    )
    held = [one["Key"] for page in listed for one in page.get("Contents", [])]
    wanted = [key for key in held if names is None or key[len(prefix) :] in names]
    if not wanted:
        print(f"nothing is published as {name}, so this starts from none", flush=True)
        return
    if names is None:
        shutil.rmtree(into, ignore_errors=True)
    files = [into / key[len(prefix) :] for key in wanted]
    for one in files:
        one.parent.mkdir(parents=True, exist_ok=True)
    print(f"fetching {name}, {len(wanted):,} files", flush=True)
    with cancellable_pool(ThreadPoolExecutor(downloads)) as fetching:
        list(
            fetching.map(client.download_file, repeat(bucket), wanted, map(str, files))
        )


def download_verdicts(project) -> None:
    """Bring both reviews down, beside the version's analysis."""
    names = [path.name for path in VERDICTS]
    download_folder(project, "verdicts", analysis_paths.VERSION_ANALYSIS_ROOT, names)


def download_manifest(project) -> None:
    """Bring the training build's manifest down, where the notebooks read it."""
    name = training_settings().name
    download_folder(
        project,
        f"dataset-{name}",
        building_paths.dataset_root(name),
        [building_paths.DATASET_MANIFEST_NAME],
    )


def upload_verdicts(project):
    """Publish the reviews, the one file each annotation notebook writes.

    Args:
        project: The DigitalHub project to log the reviews into.

    Returns:
        artifact: The logged artifact.
    """
    written = [path for path in VERDICTS if path.is_file()]
    return upload_folder(
        project, "verdicts", analysis_paths.VERSION_ANALYSIS_ROOT, written
    )
