#!/usr/bin/env python
"""The dataset build: run here by default, or submitted with --dh."""

from __future__ import annotations

import argparse
import os
import time

from dhub import archives, configs, submit
from digitalhub_runtime_python import handler
from rich.console import Console

import utils.disk.paths as paths
from building import console, runner, settings

BUILD_HANDLER = "scripts.building_pipeline:run_build"

_PUBLISHED = configs.load().publishes
_DATASET = _PUBLISHED["dataset"]
_SELECTION = _PUBLISHED["selection"]


def build_dataset(force: bool = False, cores: int | None = None) -> int:
    """Build the dataset the selection asks for, over as much of it as configured.

    Args:
        force: Whether to rebuild crops that are already written.
        cores: How many cores the run was given, or None for the machine's.

    Returns:
        code: A process exit code, non zero when any product failed to build.
    """
    choices = settings.load(cores=cores)
    printing = Console()
    started_at = time.monotonic()
    outcomes = runner.run_build(
        choices, printing, paths.dataset_root(choices.name), force=force
    )
    console.print_summary(outcomes, time.monotonic() - started_at, printing)
    return 1 if any(one.error for one in outcomes) else 0


@handler(outputs=[_DATASET])
def run_build(project, force: bool = False, cores: int | None = None):
    """Build the dataset on DigitalHub and publish what it left on disk.

    Args:
        project: The DigitalHub project the dataset is logged into.
        force: Whether to rebuild crops that are already written.
        cores: How many cores the run was given, as the job was sized.

    Returns:
        dataset: The published dataset, one object per crop.

    Raises:
        RuntimeError: When a product failed, which leaves the dataset short of
            what the selection asked for.
    """
    os.environ[console.PLAIN_LOG_ENV] = "1"
    choices = settings.load(cores=cores)
    # The platform clones the repo alone, so the selection comes off its archive
    print("fetching the selection", flush=True)
    archives.unpack_archive(
        project.get_artifact(_SELECTION).download(overwrite=True), paths.SELECTION_ROOT
    )
    print(f"building {choices.share:.0%} of the dataset as {choices.name}", flush=True)
    failed = build_dataset(force, cores)
    # The build's own name is carried through, so one never overwrites another
    published = archives.published_folder(
        project,
        paths.dataset_root(choices.name),
        f"{_DATASET}-{choices.name}",
        "The cropped observations and their index, one object per crop; read "
        "observations.parquet and ask the store for the crops it names.",
    )
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
        "--force", action="store_true", help="rebuild crops that are already written"
    )
    parsed.add_argument("--ref", default="main", help="branch, tag, or commit to run")
    arguments = parsed.parse_args()

    if arguments.dh:
        return submit.submitted(
            "build", BUILD_HANDLER, arguments.ref, "cores", force=arguments.force
        )
    return build_dataset(arguments.force)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        console.print_interrupted()
        raise SystemExit(130) from None
