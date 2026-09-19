#!/usr/bin/env python
"""The training build: run here by default, or submitted with --dh."""

from __future__ import annotations

import argparse
import os

from dhub import archives, build, submit
from dhub import configs as platform
from digitalhub_runtime_python import handler

from analysis import paths as analysis_paths
from analysis.labels import artifacts
from analysis.selector.models.selection import Selection
from analysis.utils import dataset_list
from common.building import build as building
from common.config import load_config
from common.console import PLAIN_LOG_ENV, print_interrupted
from training import paths
from training.building import draw
from training.building.models.settings import Settings

BUILD_HANDLER = "scripts.build_training:run_build"

_PUBLISHED = platform.load().publishes
_DATASET = _PUBLISHED["dataset"]
_SELECTION = _PUBLISHED["selection"]
_LABELS = _PUBLISHED["labels"]


def training_selections(settings: Settings) -> list[Selection]:
    """Read the selection and draw the tiles the training build covers.

    Args:
        settings: The settled choices for the build, which size the draw.

    Returns:
        picked: The tiles to build, each with the observations its window keeps,
            and none the evaluation set holds.

    Raises:
        FileNotFoundError: When the analysis pipeline has written no labels, since
            training would then take the tiles evaluation is meant to hold out.
    """
    held_out = {one.tile for one in artifacts.read_labels() if one.drawn}
    return draw.drawn_selections(dataset_list.read_dataset_list(), settings, held_out)


@handler(outputs=[_DATASET])
def run_build(project, force: bool = False, workers: int | None = None):
    """Build the training dataset on DigitalHub and publish what it left on disk.

    Args:
        project: The DigitalHub project the dataset is logged into.
        force: Whether to build the dataset again from nothing.
        workers: How many products to build at once, as the job was sized.

    Returns:
        dataset: The published dataset, one object per crop.
    """
    os.environ[PLAIN_LOG_ENV] = "1"
    # The platform clones the repo alone, so both come off their archives
    print("fetching the selection and the evaluation labels", flush=True)
    archives.unpack_archive(project, _SELECTION, analysis_paths.SELECTION_ROOT)
    archives.unpack_archive(project, _LABELS, analysis_paths.LABELS_ROOT)
    settings = load_config(paths.BUILDING_CONFIG_PATH, Settings, workers=workers)
    return build.published_dataset(
        project, settings, training_selections(settings), force
    )


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
    settings = load_config(paths.BUILDING_CONFIG_PATH, Settings)
    return building.build_dataset(
        settings, training_selections(settings), arguments.force
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print_interrupted("written crops")
        raise SystemExit(130) from None
