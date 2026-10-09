#!/usr/bin/env python
"""The training build: run here by default, or submitted with --dh."""

from __future__ import annotations

from dhub import build, store
from digitalhub_runtime_python import handler

from analysis.ground_truth import catalogue
from analysis.selector.artifacts import read_selection
from analysis.selector.models.selection import Selection
from building import draw
from building.models.settings import TrainingSettings
from common.config import training_settings


def training_selections(settings: TrainingSettings) -> list[Selection]:
    """Draw the training tiles, none held out or refused, failing without labels."""
    return draw.draw_training(
        read_selection(),
        settings,
        catalogue.read_labels(),
        catalogue.read_refused(),
    )


@handler(outputs=["dataset"])
def run_build(project, force: bool = False, workers: int | None = None, overrides=()):
    """Build the training dataset on DigitalHub and publish what it left on disk.

    Args:
        project: The DigitalHub project the dataset is logged into.
        force: Whether to build from nothing rather than fill in what is missing.
        workers: How many products to build at once, as the job was sized.
        overrides: Hydra overrides of the build's config file.

    Returns:
        dataset: The published dataset, one object per crop.
    """
    store.download_verdicts(project)
    return build.published_build(
        project,
        training_settings(workers, overrides),
        training_selections,
        ("selection", "labels"),
        force,
    )


def main() -> int:
    """Run the build where it was asked for, over as much as it was asked for.

    Returns:
        code: A process exit code, non zero when a product failed.
    """
    return build.build_exit_code(
        __doc__, "build_training", training_settings, training_selections
    )


if __name__ == "__main__":
    raise SystemExit(main())
