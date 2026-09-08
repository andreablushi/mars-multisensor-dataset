"""The archives a platform run publishes, and the ones it reads back."""

from __future__ import annotations

import shutil
import tarfile
from pathlib import Path

from digitalhub.utils.exceptions import BackendError

from shared import paths


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
    packed = Path(
        shutil.make_archive(
            str(paths.DATA_ROOT / name),
            "gztar",
            root_dir=root.parent,
            base_dir=root.name,
        )
    )
    print(f"uploading {name}, {packed.stat().st_size / 1e6:.0f} MB", flush=True)
    try:
        return project.log_artifact(
            name=name, kind="artifact", source=str(packed), description=description
        )
    finally:
        # The platform holds it now, so the job keeps neither file nor pages
        packed.unlink(missing_ok=True)


def published_folder(project, root: Path, name: str, description: str):
    """Publish one tree file by file, each addressable where it lands.

    Args:
        project: The DigitalHub project to log the folder into.
        root: The directory to publish, whose files keep the paths they hold
            inside it, which is what the index names them by.
        name: The name the folder is published under.
        description: What the folder holds, and how it is read.

    Returns:
        artifact: The logged artifact.
    """
    files = [one for one in root.rglob("*") if one.is_file()]
    held = sum(one.stat().st_size for one in files)
    print(f"uploading {name}, {len(files):,} files, {held / 1e6:.0f} MB", flush=True)
    return project.log_artifact(
        name=name, kind="artifact", source=str(root), description=description
    )


def download_folder(project, name: str, into: Path) -> int:
    """Put a published folder back where a run reads it, so it fills in the rest.

    Args:
        project: The DigitalHub project the folder was logged into.
        name: The name the folder was published under, which need not be published
            yet: a first run has nothing to fill in from.
        into: The directory it fills, keeping whatever is already there.

    Returns:
        files: How many files it now holds, and zero where nothing is published.
    """
    try:
        artifact = project.get_artifact(name)
    # A name nothing is published under leaves the platform with no version to
    # hand back, which it reports as a plain backend error and not a missing one.
    except BackendError:
        print(f"nothing is published as {name}, so this starts from none", flush=True)
        return 0
    into.mkdir(parents=True, exist_ok=True)
    artifact.download(str(into), overwrite=True)
    files = sum(1 for one in into.rglob("*") if one.is_file())
    print(f"filling in from {name}, {files:,} files already built", flush=True)
    return files


def unpack_archive(downloaded: str, into: Path) -> None:
    """Put a published archive back where the pipeline reads it, and nothing else.

    Args:
        downloaded: The archive the platform left, or the directory holding it.
        into: The directory the archive fills, emptied first so that what it
            holds afterwards is what was published and only that.

    Raises:
        RuntimeError: When the download left no archive to unpack.
    """
    path = Path(downloaded)
    if path.is_dir():
        found = sorted(path.glob("*.tar.gz"))
        if not found:
            raise RuntimeError(f"no archive was downloaded into {path}")
        path = found[0]
    shutil.rmtree(into, ignore_errors=True)
    into.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path) as packed:
        packed.extractall(into.parent, filter="data")
