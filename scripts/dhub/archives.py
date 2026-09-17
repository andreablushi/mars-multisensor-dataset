"""The archives a platform run publishes, and the ones it reads back."""

from __future__ import annotations

import shutil
import tarfile
import warnings
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

from digitalhub import get_s3_client
from digitalhub.stores.data.api import get_default_store

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


def stored_folder(project, name: str) -> tuple[object, str, str]:
    """Return what one published folder is reached through, with fresh credentials.

    Args:
        project: The DigitalHub project the folder belongs to.
        name: The name the folder is published under.

    Returns:
        client: The S3 client the store is asked through.
        bucket: The bucket the folder is kept in.
        prefix: The key every object of the folder starts with.
    """
    credentials.refresh()
    place = urlparse(published_at(project, name, ""))
    return get_s3_client(), place.netloc, place.path.lstrip("/")


def published_folder(
    project,
    root: Path,
    files: Sequence[Path],
    last: Sequence[Path],
    name: str,
    description: str,
    uploads: int,
):
    """Publish some files of one tree, sending many at once, and a few after them.

    Args:
        project: The DigitalHub project to log the folder into.
        root: The directory they sit in, whose paths inside it they keep, which
            is what the index names them by.
        files: What to send, in any order.
        last: What is sent one by one, only once every file is up.
        name: The name the folder is published under.
        description: What the folder holds, and how it is read.
        uploads: How many files are sent at once.

    Returns:
        artifact: The logged artifact.
    """
    client, bucket, prefix = stored_folder(project, name)
    going = sum(path.stat().st_size for path in [*files, *last])
    told = f"{len(files) + len(last):,} files, {going / 1e6:.0f} MB"
    print(f"uploading {name}, {told}", flush=True)

    def send(path: Path) -> None:
        """Send one file to the key its path inside the tree names.

        Args:
            path: The file to send.
        """
        key = prefix + path.relative_to(root).as_posix()
        client.upload_file(Filename=str(path), Bucket=bucket, Key=key)

    with ThreadPoolExecutor(max_workers=uploads) as sending:
        list(sending.map(send, files))
    for path in last:
        send(path)
    return project.new_artifact(
        name=name,
        kind="artifact",
        path=published_at(project, name, ""),
        description=description,
    )


def download_files(project, name: str, into: Path, names: Sequence[str]) -> None:
    """Put the named files of a published folder back where a run reads them.

    Args:
        project: The DigitalHub project the folder was logged into.
        name: The name the folder was published under, which need not be published
            yet: a first run has nothing to fill in from.
        into: The directory they land in, keeping whatever is already there.
        names: The files to bring down, each at the top of the folder.
    """
    client, bucket, prefix = stored_folder(project, name)
    listed = client.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter="/")
    held = {one["Key"].removeprefix(prefix) for one in listed.get("Contents", [])}
    wanted = [one for one in names if one in held]
    if not wanted:
        print(f"nothing is published as {name}, so this starts from none", flush=True)
        return
    into.mkdir(parents=True, exist_ok=True)
    for one in wanted:
        client.download_file(Bucket=bucket, Key=prefix + one, Filename=str(into / one))
    print(f"filling in from {name}, {len(wanted):,} files of its index", flush=True)


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
