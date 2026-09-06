#!/usr/bin/env python
"""The dataset build: run here by default, or submitted with --dh."""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Callable

from dhub import archives, configs
from rich.console import Console

import utils.disk.paths as paths
from building import console, runner, settings

try:
    from digitalhub_runtime_python import handler
except ModuleNotFoundError:
    # Only a submitted run installs the platform, so here the mark does nothing
    def handler(outputs: list[str]) -> Callable:
        """Leave a handler as it is when the platform is not installed."""
        return lambda called: called


BUILD_HANDLER = "scripts.building_pipeline:run_build"

_DATASET = configs.load().publishes.get("dataset", "dataset")


def _published(name: str) -> str:
    """Return what one build of the dataset is published under.

    Args:
        name: What the build is called, as its config names it.

    Returns:
        The artifact name, which carries the build's own so one never
        overwrites another.
    """
    return f"{_DATASET}-{name}"


def build(force: bool = False, workers: int | None = None) -> int:
    """Build the dataset the selection asks for, over as much of it as configured.

    Args:
        force: Whether to rebuild crops that are already written.
        workers: How many jobs to run at once, or None for the config.

    Returns:
        A process exit code, non zero when any product failed to build.
    """
    choices = settings.load(workers=workers)
    printing = Console()
    started_at = time.monotonic()
    outcomes = runner.run_build(
        choices, printing, paths.dataset_root(choices.name), force=force
    )
    console.print_summary(outcomes, time.monotonic() - started_at, printing)
    return 1 if any(one.failed for one in outcomes) else 0


@handler(outputs=[_DATASET])
def run_build(project, force: bool = False, workers: int | None = None):
    """Build the dataset on DigitalHub and publish what it left on disk.

    Args:
        project: The DigitalHub project the archive is logged into.
        force: Whether to rebuild crops that are already written.
        workers: How many jobs to run at once, as the job was sized.

    Returns:
        The uploaded archive of the dataset.

    Raises:
        RuntimeError: When a product failed, which leaves the dataset short of
            what the selection asked for.
    """
    os.environ[console.PLAIN_LOG_ENV] = "1"
    choices = settings.load(workers=workers)
    print(f"building {choices.share:.0%} of the dataset as {choices.name}", flush=True)
    failed = build(force, workers)
    published = archives.logged(
        project,
        paths.dataset_root(choices.name),
        _published(choices.name),
        "The cropped observations and their index; unpack under "
        "data/building/dataset/.",
    )
    if failed:
        raise RuntimeError("the build had failures; the archive holds what finished")
    print("done", flush=True)
    return published


def main() -> int:
    """Run the build where it was asked for, over as much as it was asked for.

    Returns:
        A process exit code, non zero when a product failed or an image did not
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
        # Only a submission needs the platform installed, so it is asked for here
        from dhub import submit

        return submit.submitted(
            "build", BUILD_HANDLER, arguments.ref, force=arguments.force
        )
    return build(arguments.force)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        console.print_interrupted()
        raise SystemExit(130) from None
