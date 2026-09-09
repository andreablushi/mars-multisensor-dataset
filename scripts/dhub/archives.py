"""The archives a platform run publishes, and the ones it reads back."""

from __future__ import annotations

import shutil
import tarfile
import warnings
from pathlib import Path
from urllib.parse import urlparse

from digitalhub import get_s3_client
from digitalhub.stores.data.api import get_default_store
from digitalhub.utils.exceptions import BackendError

from dhub import credentials
from shared import paths

ANALYSIS_DIR = "analysis"

# The platform says twice per publish that 0.16 renames what it is called by.
warnings.filterwarnings("ignore", ".*0\\.16", UserWarning)


def published_at(project, *parts: str) -> str:
    """Return where one published thing belongs, in the store the platform uses.

    Args:
        project: The DigitalHub project, which names its own tree in the store.
        *parts: The directories it is gathered into and the name it lands as,
            in order, an empty last part leaving it a directory of its own.

    Returns:
        path: The destination, which the platform writes to as given rather than
            generating one of its own per version.
    """
    root = get_default_store(project.name)
    return "/".join((root, project.name, "artifacts", *parts))


def published_archive(project, root: Path, name: str, description: str):
    """Pack one tree and publish it as a single archive, saying how big it went up.

    Args:
        project: The DigitalHub project to log the archive into.
        root: The directory to pack, whose name the archive entries carry.
        name: The name the archive is published under.
        description: What the archive holds, and where it unpacks.

    Returns:
        artifact: The logged artifact.
    """
    credentials.refresh()
    made = shutil.make_archive(
        str(paths.DATA_ROOT / name), "gztar", root.parent, root.name
    )
    packed = Path(made)
    print(f"uploading {name}, {packed.stat().st_size / 1e6:.0f} MB", flush=True)
    try:
        return project.log_artifact(
            name=name,
            kind="artifact",
            source=str(packed),
            path=published_at(project, ANALYSIS_DIR, packed.name),
            description=description,
        )
    finally:
        # The platform holds it now, so the job keeps neither file nor pages
        packed.unlink(missing_ok=True)


def published_folder(
    project, root: Path, name: str, description: str, written_once: str
):
    """Publish one tree file by file, sending only what the store does not hold.

    Args:
        project: The DigitalHub project to log the folder into.
        root: The directory to publish, whose files keep the paths they hold
            inside it, which is what the index names them by.
        name: The name the folder is published under.
        description: What the folder holds, and how it is read.
        written_once: The suffix of the files a build writes once and never
            again. One of those the store holds at the size it was written is
            left where it is; everything else goes up every time, the index
            being rewritten at every checkpoint.

    Returns:
        artifact: The logged artifact.
    """
    credentials.refresh()
    destination = published_at(project, name, "")
    bucket = urlparse(destination).netloc
    prefix = urlparse(destination).path.lstrip("/")
    client = get_s3_client()

    held = {}
    pages = client.get_paginator("list_objects_v2")
    for page in pages.paginate(Bucket=bucket, Prefix=prefix):
        for one in page.get("Contents", []):
            held[one["Key"]] = one["Size"]

    files = [one for one in root.rglob("*") if one.is_file()]
    sending = []
    for path in files:
        key = prefix + path.relative_to(root).as_posix()
        if path.suffix == written_once and held.get(key) == path.stat().st_size:
            continue
        sending.append((path, key))

    going = sum(path.stat().st_size for path, _ in sending)
    told = f"{len(sending):,} of {len(files):,} files, {going / 1e6:.0f} MB"
    print(f"uploading {name}, {told}", flush=True)
    for path, key in sending:
        client.upload_file(Filename=str(path), Bucket=bucket, Key=key)

    return project.new_artifact(
        name=name,
        kind="artifact",
        path=destination,
        description=description,
    )


def download_folder(project, name: str, into: Path) -> None:
    """Put a published folder back where a run reads it, so it fills in the rest.

    Args:
        project: The DigitalHub project the folder was logged into.
        name: The name the folder was published under, which need not be published
            yet: a first run has nothing to fill in from.
        into: The directory it fills, keeping whatever is already there.
    """
    try:
        artifact = project.get_artifact(name)
    # A name nothing is published under leaves the platform with no version to
    # hand back, which it reports as a plain backend error and not a missing one.
    except BackendError:
        print(f"nothing is published as {name}, so this starts from none", flush=True)
        return
    into.mkdir(parents=True, exist_ok=True)
    artifact.download(str(into), overwrite=True)
    files = sum(1 for one in into.rglob("*") if one.is_file())
    print(f"filling in from {name}, {files:,} files already built", flush=True)


def unpack_archive(downloaded: str, into: Path) -> None:
    """Put a published archive back where the pipeline reads it, and nothing else.

    Args:
        downloaded: The archive the platform left, which is the one file it holds.
        into: The directory the archive fills, emptied first so that what it
            holds afterwards is what was published and only that.
    """
    shutil.rmtree(into, ignore_errors=True)
    into.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(downloaded) as packed:
        packed.extractall(into.parent, filter="data")
