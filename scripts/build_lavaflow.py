#!/usr/bin/env python
"""The lava flow build: the reviewed crops for the MAE, or every candidate to review.

By default it builds from nothing the crops reviewed as reflector or none into
`data/<version>/dataset/lavaflow`, so an unsure or unreviewed crop is never fetched.
With --annotation it fills in every candidate crop into
`data/analysis/lavaflow_candidates`, for the review, or with --force deletes what it
held, on disk or published, and builds it from nothing. Run here, or submitted with
--dh.
"""

from __future__ import annotations

import shutil
from functools import partial
from pathlib import Path

from dhub import build, store, submit
from digitalhub_runtime_python import handler

from mars_multisensor_dataset.analysis import paths as analysis_paths
from mars_multisensor_dataset.analysis.ground_truth.lavaflow.labels import (
    judged_labels,
    read_lavaflow_labels,
    read_lavaflow_verdicts,
    write_lavaflow_labels,
)
from mars_multisensor_dataset.analysis.ground_truth.lavaflow.models import (
    NONE,
    REFLECTOR,
)
from mars_multisensor_dataset.analysis.selector.artifacts import read_selection
from mars_multisensor_dataset.analysis.selector.models.selection import Selection
from mars_multisensor_dataset.building import draw, paths, runner
from mars_multisensor_dataset.building.models.settings import BuildSettings
from mars_multisensor_dataset.common.config import lavaflow_settings

KEPT_VERDICTS = (REFLECTOR, NONE)
CANDIDATES_NAME = "lavaflow_candidates"


def lavaflow_root(settings: BuildSettings, annotation: bool) -> Path:
    """Return where the build is written, the candidates or the reviewed crops."""
    if annotation:
        return analysis_paths.LAVAFLOW_CANDIDATES_ROOT
    return paths.dataset_root(settings.name)


def lavaflow_selections(settings: BuildSettings, annotation: bool) -> list[Selection]:
    """Write the lava flow labels beside the build, and read the tiles they lie on.

    Args:
        settings: The settled choices for the build, which name its directory.
        annotation: Whether every candidate is built, or only the reviewed crops.

    Returns:
        picked: The tiles to build, each with the tracks labelled on it.

    Raises:
        FileNotFoundError: When the analysis pipeline has listed no lava flow crops.
    """
    labels = judged_labels(read_lavaflow_labels(), read_lavaflow_verdicts())
    if not annotation:
        labels = [one for one in labels if one.verdict in KEPT_VERDICTS]
    write_lavaflow_labels(labels, lavaflow_root(settings, annotation))
    return draw.draw_lavaflow(read_selection(), labels)


@handler(outputs=["dataset"])
def run_build(
    project,
    force: bool = False,
    workers: int | None = None,
    overrides=(),
    annotation: bool = False,
):
    """Build the lava flow dataset on DigitalHub and publish what it left on disk.

    Args:
        project: The DigitalHub project the dataset is logged into.
        force: Whether to delete the candidates and build them from nothing.
        workers: How many products to build at once, as the job was sized.
        overrides: Hydra overrides of the build's config file.
        annotation: Whether every candidate is built, or only the reviewed crops.

    Returns:
        dataset: The published dataset, one object per crop.
    """
    store.download_verdicts(project)
    settings = lavaflow_settings(workers, overrides)
    name = CANDIDATES_NAME if annotation else f"dataset-{settings.name}"
    fresh = force or not annotation
    if fresh:
        store.delete_folder(project, name)
    return build.published_build(
        project,
        settings,
        partial(lavaflow_selections, annotation=annotation),
        ("selection", "labels"),
        fresh,
        lavaflow_root(settings, annotation),
        name,
    )


def main() -> int:
    """Run the build where it was asked for.

    Returns:
        code: A process exit code, non zero when a product failed.
    """
    parser = submit.script_parser(__doc__)
    parser.add_argument(
        "--annotation",
        action="store_true",
        help="build every candidate crop to review, rather than the reviewed ones",
    )
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, as key=value")
    arguments = parser.parse_args()
    if arguments.dh:
        return submit.submitted(
            "build_lavaflow",
            arguments.ref,
            arguments.overrides,
            force=arguments.force,
            annotation=arguments.annotation,
        )
    settings = lavaflow_settings(None, arguments.overrides)
    root = lavaflow_root(settings, arguments.annotation)
    fresh = arguments.force or not arguments.annotation
    if fresh:
        shutil.rmtree(root, ignore_errors=True)
    return runner.build_dataset(
        settings,
        lavaflow_selections(settings, arguments.annotation),
        fresh,
        root=root,
    )


if __name__ == "__main__":
    raise SystemExit(main())
