#!/usr/bin/env python
"""The evaluation build: run here by default, or submitted with --dh."""

from __future__ import annotations

from dhub import build
from digitalhub_runtime_python import handler

from analysis.ground_truth import artifacts, catalogue
from analysis.selector.artifacts import read_selection
from analysis.selector.models.selection import Selection
from building import draw, paths
from building.models.settings import BuildSettings
from common.config import evaluation_settings


def evaluation_selections(settings: BuildSettings) -> list[Selection]:
    """Write the drawn labels beside the evaluation build, and read what it covers.

    Args:
        settings: The settled choices for the build, which name its directory.

    Returns:
        picked: The tiles to build, each with the observations its window keeps.

    Raises:
        FileNotFoundError: When the analysis pipeline has written no labels.
    """
    labels = catalogue.read_labels()
    root = paths.dataset_root(settings.name)
    artifacts.write_labels([one for one in labels if one.drawn], root)
    return draw.draw_evaluation(read_selection(), labels)


@handler(outputs=["dataset"])
def run_build(project, force: bool = False, workers: int | None = None, overrides=()):
    """Build the evaluation dataset on DigitalHub and publish what it left on disk.

    Args:
        project: The DigitalHub project the dataset is logged into.
        force: Whether to build from nothing rather than fill in what is missing.
        workers: How many products to build at once, as the job was sized.
        overrides: Hydra overrides of the build's config file.

    Returns:
        dataset: The published dataset, one object per crop.
    """
    return build.published_build(
        project,
        evaluation_settings(workers, overrides),
        evaluation_selections,
        ("selection", "labels"),
        force,
    )


def main() -> int:
    """Run the build where it was asked for.

    Returns:
        code: A process exit code, non zero when a product failed.
    """
    return build.build_exit_code(
        __doc__, "build_evaluation", evaluation_settings, evaluation_selections
    )


if __name__ == "__main__":
    raise SystemExit(main())
