#!/usr/bin/env python
"""The training build: run here by default, or submitted with --dh."""

from __future__ import annotations

import argparse
import os

from dhub import archives, build, submit
from dhub import configs as platform
from digitalhub_runtime_python import handler

from analysis import paths as analysis_paths
from analysis.ground_truth import artifacts
from analysis.selector.models.selection import Selection
from analysis.utils import dataset_list
from building import draw, paths
from building.build import build_dataset
from building.models.settings import TrainingSettings
from common.config import load_config
from common.console import PLAIN_LOG_ENV, print_interrupted

BUILD_HANDLER = "scripts.build_training:run_build"

_PUBLISHED = platform.load().publishes
_DATASET = _PUBLISHED["dataset"]
_SELECTION = _PUBLISHED["selection"]
_LABELS = _PUBLISHED["labels"]
_VERDICTS = _PUBLISHED["verdicts"]


def training_selections(settings: TrainingSettings) -> list[Selection]:
    """Read the selection and draw the tiles the training build covers.

    Args:
        settings: The settled choices for the build, which size the draw.

    Returns:
        picked: The tiles to build with their windows, none held out or refused.

    Raises:
        FileNotFoundError: When no labels were written, so none can be held out.
    """
    return draw.draw_training(
        dataset_list.read_dataset_list(),
        settings,
        artifacts.read_labels(),
        artifacts.read_refused(),
    )


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
    # The platform clones the repo alone, so all three come off the platform
    print("fetching the selection, the evaluation labels and the verdicts", flush=True)
    archives.unpack_archive(project, _SELECTION, analysis_paths.SELECTION_ROOT)
    archives.unpack_archive(project, _LABELS, analysis_paths.LABELS_ROOT)
    archives.download_file(project, _VERDICTS, analysis_paths.VERDICTS_PATH)
    settings = load_config(
        paths.TRAINING_CONFIG_PATH, TrainingSettings, workers=workers
    )
    return build.published_dataset(
        project, settings, training_selections(settings), force
    )


def main() -> int:
    """Run the build where it was asked for, over as much as it was asked for.

    Returns:
        code: A process exit code, non zero when a product or an image build failed.
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
            "build_training", BUILD_HANDLER, arguments.ref, force=arguments.force
        )
    settings = load_config(paths.TRAINING_CONFIG_PATH, TrainingSettings)
    return build_dataset(settings, training_selections(settings), arguments.force)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print_interrupted("written crops")
        raise SystemExit(130) from None
