#!/usr/bin/env python
"""The lava flow build: run here by default, or submitted with --dh."""

from __future__ import annotations

from dhub import build, store
from digitalhub_runtime_python import handler

from mars_multisensor_dataset.analysis.ground_truth.lavaflow.labels import (
    judged_labels,
    read_lavaflow_labels,
    read_lavaflow_verdicts,
    write_lavaflow_labels,
)
from mars_multisensor_dataset.analysis.selector.artifacts import read_selection
from mars_multisensor_dataset.analysis.selector.models.selection import Selection
from mars_multisensor_dataset.building import draw, paths
from mars_multisensor_dataset.building.models.settings import BuildSettings
from mars_multisensor_dataset.common.config import lavaflow_settings


def lavaflow_selections(settings: BuildSettings) -> list[Selection]:
    """Write the lava flow labels beside the build as reviewed so far, and read it.

    Args:
        settings: The settled choices for the build, which name its directory.

    Returns:
        picked: The tiles to build, each with the tracks labelled on it.

    Raises:
        FileNotFoundError: When the analysis pipeline has listed no lava flow crops.
    """
    labels = judged_labels(read_lavaflow_labels(), read_lavaflow_verdicts())
    write_lavaflow_labels(labels, paths.dataset_root(settings.name))
    return draw.draw_lavaflow(read_selection(), labels)


@handler(outputs=["dataset"])
def run_build(project, force: bool = False, workers: int | None = None, overrides=()):
    """Build the lava flow dataset on DigitalHub and publish what it left on disk.

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
        lavaflow_settings(workers, overrides),
        lavaflow_selections,
        ("selection", "labels"),
        force,
    )


def main() -> int:
    """Run the build where it was asked for.

    Returns:
        code: A process exit code, non zero when a product failed.
    """
    return build.build_exit_code(
        __doc__, "build_lavaflow", lavaflow_settings, lavaflow_selections
    )


if __name__ == "__main__":
    raise SystemExit(main())
