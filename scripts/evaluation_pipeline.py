#!/usr/bin/env python
"""The evaluation labels: run here by default, or submitted with --dh."""

from __future__ import annotations

import argparse
import os
from collections import Counter

from dhub import archives, submit
from dhub import configs as platform
from digitalhub_runtime_python import handler

from common.analysis import paths as analysis_paths
from common.analysis.utils import dataset_list
from common.console import PLAIN_LOG_ENV, print_interrupted
from evaluation import paths
from evaluation.analysis import artifacts, configs, draw, fetch, label
from evaluation.analysis.models.label import Label

LABELS_HANDLER = "scripts.evaluation_pipeline:run_labels"

_PUBLISHED = platform.load().publishes
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
    os.environ[PLAIN_LOG_ENV] = "1"
    print("fetching the selection", flush=True)
    archives.unpack_archive(project, _SELECTION, analysis_paths.SELECTION_ROOT)
    compute_labels(force)
    return archives.published_archive(
        project,
        paths.LABELS_ROOT,
        _LABELS,
        "Every labelled tile, the drawn ones marked; unpack under data/evaluation/.",
    )


def main() -> int:
    """Run the labelling where it was asked for.

    Returns:
        code: A process exit code, non zero when an image did not build.
    """
    parsed = argparse.ArgumentParser(description=__doc__)
    parsed.add_argument(
        "--dh", action="store_true", help="submit to DigitalHub instead of running here"
    )
    parsed.add_argument(
        "--force", action="store_true", help="fetch the feature catalogue again"
    )
    parsed.add_argument("--ref", default="main", help="branch, tag, or commit to run")
    arguments = parsed.parse_args()

    if arguments.dh:
        return submit.submitted(
            "labels", LABELS_HANDLER, arguments.ref, force=arguments.force
        )
    compute_labels(arguments.force)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print_interrupted("written labels")
        raise SystemExit(130) from None
