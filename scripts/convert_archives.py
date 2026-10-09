#!/usr/bin/env python
"""Republish every analysis archive as a folder on DigitalHub, once, with --dh."""

from __future__ import annotations

import tarfile
from pathlib import Path
from urllib.parse import urlparse

from dhub import store, submit
from digitalhub_runtime_python import handler

from common.paths import DATA_ROOT


@handler(outputs=list(store.ARTIFACTS))
def run_conversion(project, workers: int | None = None):
    """Unpack each published archive, publish it as a folder, then delete the archive.

    Args:
        project: The DigitalHub project the archives were logged into.
        workers: Unused, handed to every stage as the job was sized.

    Returns:
        folders: The folder each archive is now published as.
    """
    folders = []
    for name, root in store.ARTIFACTS.items():
        packed = project.get_artifact(name)
        held = Path(packed.download(destination=str(DATA_ROOT), overwrite=True))
        with tarfile.open(held) as archive:
            archive.extractall(root.parent, filter="data")
        held.unlink()
        folders.append(store.upload_folder(project, name, root))
        # The quota has little room, so the archive goes once its folder is up
        client, bucket, _ = store.stored_folder(project, name)
        client.delete_object(Bucket=bucket, Key=urlparse(packed.spec.path).path[1:])
        print(f"converted {name}", flush=True)
    return tuple(folders)


if __name__ == "__main__":
    raise SystemExit(
        submit.submitted("convert", submit.script_parser(__doc__).parse_args().ref)
    )
