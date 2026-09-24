#!/usr/bin/env python
"""The evaluation build: run here by default, or submitted with --dh."""

from __future__ import annotations

from dhub import args, checkpoint, submit
from dhub.artifacts import Artifact
from digitalhub_runtime_python import handler

from analysis.ground_truth import artifacts
from analysis.selector.models.selection import Selection
from analysis.utils import dataset_list
from building import draw, paths
from building.build import build_dataset
from building.configs import run
from building.models.settings import Settings

BUILD_HANDLER = "scripts.build_evaluation:run_build"
FETCHED = (Artifact.SELECTION, Artifact.LABELS)


def evaluation_selections(settings: Settings) -> list[Selection]:
    """Write the drawn labels beside the evaluation build, and read what it covers.

    Args:
        settings: The settled choices for the build, which name its directory.

    Returns:
        picked: The tiles to build, each with the observations its window keeps.

    Raises:
        FileNotFoundError: When the analysis pipeline has written no labels.
    """
    labels = artifacts.read_labels()
    root = paths.dataset_root(settings.name)
    artifacts.write_labels([one for one in labels if one.drawn], root)
    return draw.draw_evaluation(dataset_list.read_dataset_list(), labels)


@handler(outputs=[Artifact.DATASET.published])
def run_build(project, force: bool = False, workers: int | None = None):
    """Build the evaluation dataset on DigitalHub and publish what it left on disk.

    Args:
        project: The DigitalHub project the dataset is logged into.
        force: Whether to build the dataset again from nothing.
        workers: How many products to build at once, as the job was sized.

    Returns:
        dataset: The published dataset, one object per crop.
    """
    return checkpoint.published_dataset(
        project, run.evaluation_settings(workers), evaluation_selections, FETCHED, force
    )


def main() -> int:
    """Run the build where it was asked for.

    Returns:
        code: A process exit code, non zero when a product or an image build failed.
    """
    arguments = args.script_parser(__doc__).parse_args()

    if arguments.dh:
        return submit.submitted(
            "build_evaluation", BUILD_HANDLER, arguments.ref, force=arguments.force
        )
    settings = run.evaluation_settings()
    return build_dataset(settings, evaluation_selections(settings), arguments.force)


if __name__ == "__main__":
    args.run_script(main, "written crops")
