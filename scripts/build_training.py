#!/usr/bin/env python
"""The training build: run here by default, or submitted with --dh."""

from __future__ import annotations

from dhub import args, checkpoint, submit
from dhub.paths import Artifact, Function

from analysis.ground_truth import catalogue
from analysis.selector.artifacts import read_selection
from analysis.selector.models.selection import Selection
from building import draw
from building.models.settings import TrainingSettings
from building.runner import build_dataset
from common.config import training_settings


def training_selections(settings: TrainingSettings) -> list[Selection]:
    """Draw the training tiles, none held out or refused, failing without labels."""
    return draw.draw_training(
        read_selection(),
        settings,
        catalogue.read_labels(),
        catalogue.read_refused(),
    )


run_build = checkpoint.build_handler(
    training_settings,
    training_selections,
    (Artifact.SELECTION, Artifact.LABELS, Artifact.VERDICTS),
)


def main() -> int:
    """Run the build where it was asked for, over as much as it was asked for.

    Returns:
        code: A process exit code, non zero when a product failed.
    """
    parser = args.script_parser(__doc__)
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, as key=value")
    arguments = parser.parse_args()

    if arguments.dh:
        return submit.submitted(
            Function.BUILD_TRAINING,
            arguments.ref,
            arguments.overrides,
            force=arguments.force,
        )
    settings = training_settings(overrides=arguments.overrides)
    return build_dataset(settings, training_selections(settings), arguments.force)


if __name__ == "__main__":
    args.run_script(main, "written crops")
