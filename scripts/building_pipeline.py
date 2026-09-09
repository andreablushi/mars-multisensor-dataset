#!/usr/bin/env python
"""The dataset build: run here by default, or submitted with --dh."""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Callable

from dhub import archives, credentials, submit
from dhub import configs as platform
from digitalhub_runtime_python import handler
from rich.console import Console

from analysis import paths as analysis_paths
from building import console, paths, runner
from building.configs import overall
from shared.console import PLAIN_LOG_ENV

BUILD_HANDLER = "scripts.building_pipeline:run_build"

# What the published dataset holds, said once since a checkpoint publishes it too.
DATASET_HELD = (
    "The cropped observations and their index, one object per crop; read "
    "observations.parquet and ask the store for the crops it names."
)

_PUBLISHED = platform.load().publishes
_DATASET = _PUBLISHED["dataset"]
_SELECTION = _PUBLISHED["selection"]


def build_dataset(
    force: bool = False,
    workers: int | None = None,
    checkpoint: Callable[[], None] | None = None,
) -> int:
    """Build the dataset the selection asks for, over as much of it as configured.

    Args:
        force: Whether to build every crop again, rather than only the missing ones.
        workers: How many products to build at once, or None for the config.
        checkpoint: What publishes the dataset as it stands, for a platform run
            that is resumed from what it left, and None for a run here.

    Returns:
        code: A process exit code, non zero when any product failed to build.
    """
    choices = overall.load(workers=workers)
    printing = Console()
    started_at = time.monotonic()
    outcomes = runner.run_build(
        choices,
        printing,
        paths.dataset_root(choices.name),
        force=force,
        checkpoint=checkpoint,
    )
    console.print_summary(outcomes, time.monotonic() - started_at, printing)
    return 1 if any(one.error for one in outcomes) else 0


def published_dataset(project, root, name):
    """Publish the dataset, on credentials minted for the publish itself.

    Args:
        project: The DigitalHub project the dataset is logged into.
        root: The directory holding the crops and their index.
        name: The name the dataset is published under.

    Returns:
        artifact: The logged artifact.
    """
    # The store hands back a refusal for a lapsed token as for anything else.
    if not credentials.refreshed():
        print("no credentials were minted, asking on the ones held", flush=True)
    return archives.published_folder(
        project, root, name, DATASET_HELD, paths.SAMPLE_SUFFIX
    )


@handler(outputs=[_DATASET])
def run_build(project, force: bool = False, workers: int | None = None):
    """Build the dataset on DigitalHub and publish what it left on disk.

    Args:
        project: The DigitalHub project the dataset is logged into.
        force: Whether to build the dataset again from nothing, rather than
            filling in whatever the last build of it left missing.
        workers: How many products to build at once, as the job was sized.

    Returns:
        dataset: The published dataset, one object per crop.

    Raises:
        RuntimeError: When a product failed, which leaves the dataset short of
            what the selection asked for.
    """
    os.environ[PLAIN_LOG_ENV] = "1"
    choices = overall.load(workers=workers)
    # The platform clones the repo alone, so the selection comes off its archive
    print("fetching the selection", flush=True)
    archives.unpack_archive(
        project.get_artifact(_SELECTION).download(overwrite=True),
        analysis_paths.SELECTION_ROOT,
    )
    # The build's own name is carried through, so one never overwrites another
    published_as = f"{_DATASET}-{choices.name}"
    # A job starts on an empty disk, so what is already built comes off the platform
    if not force:
        archives.download_folder(
            project, published_as, paths.dataset_root(choices.name)
        )
    print(f"building {choices.share:.0%} of the dataset as {choices.name}", flush=True)
    root = paths.dataset_root(choices.name)

    def checkpoint() -> None:
        """Publish what the build has finished, so a run that dies resumes from it."""
        published_dataset(project, root, published_as)

    failed = build_dataset(force, workers, checkpoint)
    published = published_dataset(project, root, published_as)
    if failed:
        raise RuntimeError(
            "the build had failures; what was published holds what finished"
        )
    print("done", flush=True)
    return published


def main() -> int:
    """Run the build where it was asked for, over as much as it was asked for.

    Returns:
        code: A process exit code, non zero when a product failed or an image did not
            build.
    """
    parsed = argparse.ArgumentParser(description=__doc__)
    parsed.add_argument(
        "--dh", action="store_true", help="submit to DigitalHub instead of running here"
    )
    parsed.add_argument(
        "--force",
        action="store_true",
        help="build the dataset again from nothing, rather than filling in what "
        "the last build left missing",
    )
    parsed.add_argument("--ref", default="main", help="branch, tag, or commit to run")
    arguments = parsed.parse_args()

    if arguments.dh:
        return submit.submitted(
            "build", BUILD_HANDLER, arguments.ref, force=arguments.force
        )
    return build_dataset(arguments.force)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        console.print_interrupted()
        raise SystemExit(130) from None
