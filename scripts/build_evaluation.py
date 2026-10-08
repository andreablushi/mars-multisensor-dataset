#!/usr/bin/env python
"""The evaluation build: run here by default, or submitted with --dh."""

from __future__ import annotations

from dhub import args, checkpoint, submit
from dhub.paths import Artifact, Function

from analysis.ground_truth import artifacts, catalogue
from analysis.selector.artifacts import read_selection
from analysis.selector.models.selection import Selection
from building import draw, paths
from building.models.settings import BuildSettings
from building.runner import build_dataset
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


run_build = checkpoint.build_handler(
    evaluation_settings,
    evaluation_selections,
    (Artifact.SELECTION, Artifact.LABELS),
)


def main() -> int:
    """Run the build where it was asked for.

    Returns:
        code: A process exit code, non zero when a product failed.
    """
    parser = args.script_parser(__doc__)
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, as key=value")
    arguments = parser.parse_args()

    if arguments.dh:
        return submit.submitted(
            Function.BUILD_EVALUATION,
            arguments.ref,
            arguments.overrides,
            force=arguments.force,
        )
    settings = evaluation_settings(overrides=arguments.overrides)
    return build_dataset(settings, evaluation_selections(settings), arguments.force)


if __name__ == "__main__":
    args.run_script(main, "written crops")
