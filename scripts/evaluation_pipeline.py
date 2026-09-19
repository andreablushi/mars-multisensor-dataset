#!/usr/bin/env python
"""The evaluation labels and build: run here by default, or submitted with --dh."""

from __future__ import annotations

import argparse
import os
from collections import Counter

from dhub import archives, build, submit
from dhub import configs as platform
from digitalhub_runtime_python import handler

from common.analysis import paths as analysis_paths
from common.analysis.selector.models.selection import Selection
from common.analysis.utils import dataset_list
from common.building import build as building
from common.building import paths as building_paths
from common.console import PLAIN_LOG_ENV, print_interrupted
from evaluation import paths
from evaluation.analysis import artifacts, configs, draw, fetch, label
from evaluation.analysis.models.label import Label
from evaluation.building import configs as building_configs
from evaluation.building import draw as building_draw

LABELS_HANDLER = "scripts.evaluation_pipeline:run_labels"
EVALUATION_HANDLER = "scripts.evaluation_pipeline:run_evaluation"

_PUBLISHED = platform.load().publishes
_DATASET = _PUBLISHED["dataset"]
_LABELS = _PUBLISHED["labels"]
_SELECTION = _PUBLISHED["selection"]


def compute_labels(force: bool = False) -> list[Label]:
    """Label every kept tile, draw a balanced set of them, and write both down.

    Args:
        force: Whether to fetch the feature catalogue again rather than read the
            one cached.

    Returns:
        labels: Every labelled tile, the drawn ones marked so.
    """
    settings = configs.load()
    labels = draw.drawn_labels(
        label.labelled_tiles(
            dataset_list.read_dataset_list(),
            fetch.read_features(refresh=force),
            settings,
        ),
        settings,
    )
    artifacts.write_labels(labels)
    held = Counter(one.label for one in labels)
    drawn = Counter(one.label for one in labels if one.drawn)
    for rule in settings.rules:
        print(f"{rule.label}: {drawn[rule.label]} drawn of {held[rule.label]}")
    return labels


def evaluation_selections(labels: list[Label]) -> list[Selection]:
    """Write the drawn labels beside the evaluation build, and read what it covers.

    Args:
        labels: Every labelled tile, the drawn ones marked so.

    Returns:
        picked: The tiles to build, each with the observations its window keeps.
    """
    drawn = [one for one in labels if one.drawn]
    root = building_paths.dataset_root(building_configs.load_name())
    artifacts.write_labels(drawn, root)
    return building_draw.drawn_selections(dataset_list.read_dataset_list(), drawn)


def published_labels(project, force: bool) -> tuple[list[Label], object]:
    """Label the kept tiles on DigitalHub and publish the labels.

    Args:
        project: The DigitalHub project the labels are logged into.
        force: Whether to fetch the feature catalogue again.

    Returns:
        labels: Every labelled tile, the drawn ones marked so.
        archive: The archive they were published as.
    """
    os.environ[PLAIN_LOG_ENV] = "1"
    print("fetching the selection", flush=True)
    archives.unpack_archive(project, _SELECTION, analysis_paths.SELECTION_ROOT)
    labels = compute_labels(force)
    return labels, archives.published_archive(
        project,
        paths.LABELS_ROOT,
        _LABELS,
        "Every labelled tile, the drawn ones marked; unpack under data/evaluation/.",
    )


@handler(outputs=[_LABELS])
def run_labels(project, force: bool = False, workers: int | None = None):
    """Label the kept tiles on DigitalHub and publish the labels.

    Args:
        project: The DigitalHub project the labels are logged into.
        force: Whether to fetch the feature catalogue again.
        workers: Unused, as every stage is handed the cores it was sized with.

    Returns:
        labels: The archive of every labelled tile.
    """
    return published_labels(project, force)[1]


@handler(outputs=[_LABELS, _DATASET])
def run_evaluation(project, force: bool = False, workers: int | None = None):
    """Label the kept tiles on DigitalHub, then build the drawn ones and publish both.

    Args:
        project: The DigitalHub project everything is logged into.
        force: Whether to fetch the feature catalogue again and build the
            dataset again from nothing.
        workers: How many products to build at once, as the job was sized.

    Returns:
        labels: The archive of every labelled tile.
        dataset: The published dataset, one object per crop.
    """
    labels, archive = published_labels(project, force)
    return archive, build.published_dataset(
        project,
        building_configs.load_name(),
        evaluation_selections(labels),
        force,
        workers,
    )


def main() -> int:
    """Run the labelling, and the build unless told not to, where it was asked for.

    Returns:
        code: A process exit code, non zero when a product failed or an image did not
            build.
    """
    parsed = argparse.ArgumentParser(description=__doc__)
    parsed.add_argument(
        "--dh", action="store_true", help="submit to DigitalHub instead of running here"
    )
    parsed.add_argument(
        "--only-labels",
        action="store_true",
        help="label the kept tiles and draw the set, and build nothing",
    )
    parsed.add_argument(
        "--force",
        action="store_true",
        help="fetch the feature catalogue again and build the dataset from nothing",
    )
    parsed.add_argument("--ref", default="main", help="branch, tag, or commit to run")
    arguments = parsed.parse_args()

    if arguments.dh:
        if arguments.only_labels:
            return submit.submitted(
                "labels", LABELS_HANDLER, arguments.ref, force=arguments.force
            )
        return submit.submitted(
            "evaluation", EVALUATION_HANDLER, arguments.ref, force=arguments.force
        )
    labels = compute_labels(arguments.force)
    if arguments.only_labels:
        return 0
    return building.build_dataset(
        building_configs.load_name(), evaluation_selections(labels), arguments.force
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print_interrupted("written crops")
        raise SystemExit(130) from None
