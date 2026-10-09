"""Building one dataset here, or on DigitalHub published a checkpoint at a time."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path

from analysis.selector.models.selection import Selection
from building import paths, runner
from building.models.settings import BuildSettings
from building.preprocessing.ctx.isis import install_isis
from common.console import PLAIN_LOG_ENV
from dhub import store, submit


def checkpoint(project, root: Path, name: str, uploads: int):
    """Publish the dataset as it stands, then delete the crops it sent from disk.

    Args:
        project: The DigitalHub project the dataset is logged into.
        root: The dataset's own root directory.
        name: The name the dataset is published under.
        uploads: How many crops are sent at once.

    Returns:
        dataset: The published dataset.
    """
    crops = paths.crop_paths(root)
    print(f"CHECKPOINT UPLOADING AND CLEANING {len(crops):,} crops", flush=True)
    # Whatever sits beside the crops describes them, so it goes up after them
    index = sorted(one for one in root.iterdir() if one.is_file())
    dataset = store.upload_folder(project, name, root, crops, index, uploads)
    for crop in crops:
        crop.unlink()
    return dataset


def fetch_build(
    project, workers: int, build: str, into: Path, names: Sequence[str]
) -> None:
    """Bring the named objects of one published build back into a directory."""
    store.download_folder(project, f"dataset-{build}", into, names, workers)


def published_build[T: BuildSettings](
    project,
    settings: T,
    selections: Callable[[T], list[Selection]],
    fetched: Sequence[str],
    force: bool,
):
    """Build one dataset on DigitalHub and publish what it left on disk.

    Args:
        project: The DigitalHub project the dataset is logged into.
        settings: The settled choices for the build, sized as the job was.
        selections: What reads the tiles to build once everything fetched is down.
        fetched: The artifacts the build reads, brought down first onto the job's
            empty disk.
        force: Whether to build from nothing rather than fill in what is missing.

    Returns:
        dataset: The published dataset, one object per crop.

    Raises:
        RuntimeError: When a product failed.
    """
    os.environ[PLAIN_LOG_ENV] = "1"
    # Only a stage marked isis is told ISISROOT, and its job starts without ISIS
    if "ISISROOT" in os.environ:
        install_isis()
    for one in fetched:
        store.download_folder(project, one, store.ARTIFACTS[one])
    name = f"dataset-{settings.name}"
    root = paths.dataset_root(settings.name)
    # A job starts on an empty disk, so only the index of what is built comes down
    if not force:
        store.download_folder(project, name, root, paths.INDEX_NAMES)
    print(f"building the dataset as {settings.name}", flush=True)
    published = partial(checkpoint, project, root, name, settings.workers)
    fetch = partial(fetch_build, project, settings.workers)
    failed = runner.build_dataset(
        settings, selections(settings), force, published, fetch
    )
    dataset = published()
    store.upload_folder(project, "selection", store.ARTIFACTS["selection"])
    if failed:
        raise RuntimeError(
            "the build had failures; what was published holds what finished"
        )
    print("done", flush=True)
    return dataset


def build_exit_code[T: BuildSettings](
    description: str,
    stage: str,
    settled: Callable[[int | None, Sequence[str]], T],
    selections: Callable[[T], list[Selection]],
) -> int:
    """Parse a build script's flags, then submit the build or run it here.

    Args:
        description: What the script does, shown by --help.
        stage: The stage to submit when --dh is given.
        settled: What settles the build, given its cores and the overrides it was
            started with.
        selections: What reads the tiles to build once the build is settled.

    Returns:
        code: A process exit code, non zero when a product failed.
    """
    parser = submit.script_parser(description)
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, as key=value")
    arguments = parser.parse_args()

    if arguments.dh:
        return submit.submitted(
            stage, arguments.ref, arguments.overrides, force=arguments.force
        )
    settings = settled(None, arguments.overrides)
    return runner.build_dataset(settings, selections(settings), arguments.force)
