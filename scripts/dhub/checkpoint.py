"""Building one dataset on DigitalHub, published a checkpoint at a time."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path

from digitalhub_runtime_python import handler

from analysis.selector.models.selection import Selection
from building import paths, runner
from building.models.settings import Settings
from common.console import PLAIN_LOG_ENV
from dhub import archives
from dhub.paths import Artifact


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
    dataset = archives.published_folder(project, root, crops, index, name, uploads)
    for crop in crops:
        crop.unlink()
    return dataset


def build_handler[T: Settings](
    settled: Callable[[int | None], T],
    selections: Callable[[T], list[Selection]],
    fetched: Sequence[Artifact],
) -> Callable:
    """Return the handler a build's job calls, the last checkpoint publishing it all.

    Args:
        settled: What settles the build, given the cores the job was sized with.
        selections: What reads the tiles to build once everything fetched is down.
        fetched: What the build reads, brought down first onto the job's empty disk.

    Returns:
        run_build: The handler, as the platform imports and calls it.
    """

    @handler(outputs=[Artifact.DATASET.published])
    def run_build(project, force: bool = False, workers: int | None = None):
        """Build one dataset on DigitalHub and publish what it left on disk.

        Args:
            project: The DigitalHub project the dataset is logged into.
            force: Whether to build from nothing rather than fill in what is missing.
            workers: How many products to build at once, as the job was sized.

        Returns:
            dataset: The published dataset, one object per crop.

        Raises:
            RuntimeError: When a product failed.
        """
        os.environ[PLAIN_LOG_ENV] = "1"
        for one in fetched:
            archives.download_artifact(project, one)
        settings = settled(workers)
        name = f"{Artifact.DATASET.published}-{settings.name}"
        root = paths.dataset_root(settings.name)
        # A job starts on an empty disk, so only the index of what is built comes down
        if not force:
            archives.download_files(project, name, root, paths.INDEX_NAMES)
        print(f"building the dataset as {settings.name}", flush=True)
        published = partial(checkpoint, project, root, name, settings.workers)
        failed = runner.build_dataset(settings, selections(settings), force, published)
        dataset = published()
        if failed:
            raise RuntimeError(
                "the build had failures; what was published holds what finished"
            )
        print("done", flush=True)
        return dataset

    return run_build
