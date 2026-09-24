#!/usr/bin/env python
"""The training build: run here by default, or submitted with --dh."""

from __future__ import annotations

from dhub import args, checkpoint, submit
from dhub.artifacts import Artifact
from digitalhub_runtime_python import handler

from analysis.ground_truth import artifacts
from analysis.selector.models.selection import Selection
from analysis.utils import dataset_list
from building import draw
from building.build import build_dataset
from building.configs import run
from building.models.settings import TrainingSettings

BUILD_HANDLER = "scripts.build_training:run_build"
FETCHED = (Artifact.SELECTION, Artifact.LABELS, Artifact.VERDICTS)


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


@handler(outputs=[Artifact.DATASET.published])
def run_build(project, force: bool = False, workers: int | None = None):
    """Build the training dataset on DigitalHub and publish what it left on disk.

    Args:
        project: The DigitalHub project the dataset is logged into.
        force: Whether to build the dataset again from nothing.
        workers: How many products to build at once, as the job was sized.

    Returns:
        dataset: The published dataset, one object per crop.
    """
    return checkpoint.published_dataset(
        project, run.training_settings(workers), training_selections, FETCHED, force
    )


def main() -> int:
    """Run the build where it was asked for, over as much as it was asked for.

    Returns:
        code: A process exit code, non zero when a product or an image build failed.
    """
    arguments = args.script_parser(__doc__).parse_args()

    if arguments.dh:
        return submit.submitted(
            "build_training", BUILD_HANDLER, arguments.ref, force=arguments.force
        )
    settings = run.training_settings()
    return build_dataset(settings, training_selections(settings), arguments.force)


if __name__ == "__main__":
    args.run_script(main, "written crops")
