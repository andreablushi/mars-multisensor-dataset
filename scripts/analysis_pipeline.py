#!/usr/bin/env python
"""The analysis pipeline: run here by default, or submitted with --dh."""

from __future__ import annotations

import argparse
import os
import time
from collections import Counter

from dhub import archives, submit
from dhub import configs as platform
from digitalhub_runtime_python import handler
from rich.console import Console

from analysis import configs, console, paths, planner, runner
from analysis.coverage.artifacts import index
from analysis.labels import artifacts, draw, fetch, label
from analysis.labels import configs as labelling
from analysis.labels.models.label import Label
from analysis.metadata import file_explorer
from analysis.models.progress import CoverageSummary, DownloadSummary
from analysis.selector import select
from analysis.stats.artifacts import store
from analysis.stats.dataset import aggregate, read
from analysis.utils import dataset_list
from common.console import PLAIN_LOG_ENV, print_interrupted

PIPELINE_HANDLER = "scripts.analysis_pipeline:run_pipeline"
SELECTION_HANDLER = "scripts.analysis_pipeline:run_selection"

_PUBLISHED = platform.load().publishes
_COVERAGE = _PUBLISHED["coverage"]
_METADATA = _PUBLISHED["metadata"]
_SELECTION = _PUBLISHED["selection"]
_STATS = _PUBLISHED["stats"]
_SUMMARY = _PUBLISHED["summary"]
_LABELS = _PUBLISHED["labels"]

# Where each archive is packed from and what it holds, said once since two
# handlers publish the same ones.
ARCHIVED = {
    _COVERAGE: (
        paths.COVERAGE_ROOT,
        "Coverage events and summaries; unpack under data/analysis/.",
    ),
    _METADATA: (
        paths.METADATA_ROOT,
        "The ODE records behind each measurement; unpack under data/analysis/.",
    ),
    _SELECTION: (
        paths.SELECTION_ROOT,
        "The tiles and observations the filter keeps; unpack under data/analysis/.",
    ),
    _STATS: (
        paths.STATS_ROOT,
        "What the filter left of the dataset; unpack under data/analysis/.",
    ),
    _LABELS: (
        paths.LABELS_ROOT,
        "Every labelled tile, the drawn ones marked; unpack under data/analysis/.",
    ),
}


def archived(project, name: str):
    """Publish one archive this pipeline leaves, by the name it is published under.

    Args:
        project: The DigitalHub project the archive is logged into.
        name: The name it goes up as, which is what says where it is packed from.

    Returns:
        artifact: The logged artifact.
    """
    root, held = ARCHIVED[name]
    return archives.published_archive(project, root, name, held)


def compute_coverage(force: bool = False, workers: int | None = None) -> int:
    """Download the ODE metadata still missing and measure the coverage it left.

    Args:
        force: Whether to redo finished work rather than skip it.
        workers: How many jobs each half runs at once, or None for the config.

    Returns:
        code: A process exit code, non zero when either half had a failure.
    """
    choices = configs.load(workers=workers)
    printing = Console()
    started_at = time.monotonic()
    fetched, outcomes = runner.run_pipeline(choices, printing, force)
    elapsed = time.monotonic() - started_at
    downloaded = DownloadSummary.from_outcomes(fetched, elapsed)
    computed = CoverageSummary.from_outcomes(outcomes, elapsed)
    console.print_summary(
        downloaded,
        computed,
        index.reindex(),
        planner.unfinished(file_explorer.find_sets()),
        printing,
    )
    return 1 if computed.failed or downloaded.failed else 0


def compute_labels(force: bool = False) -> list[Label]:
    """Label every kept tile, draw the balanced set held out of training, and write it.

    Args:
        force: Whether to fetch the feature catalogue again rather than read the
            one cached.

    Returns:
        labels: Every labelled tile, the drawn ones marked so.
    """
    settings = labelling.load()
    labels = draw.drawn_labels(
        label.labelled_tiles(
            dataset_list.read_selected_tiles(),
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


def compute_selection(workers: int | None = None, force: bool = False) -> None:
    """Search every measured tile under the filter, label it, and read what both left.

    Args:
        workers: How many processes to run on at once, or None for the config.
        force: Whether to fetch the feature catalogue again rather than read the
            one cached.
    """
    workers = configs.load(workers=workers).workers
    picked = select.select_dataset(workers, console.logged("selection"))
    kept = sum(1 for one in picked if one.tile.kept)
    print(f"{kept:,} of {len(picked):,} tiles earned a place", flush=True)
    # Read off the selection just written, so they never stand for an old filter
    measured = read.measure_every_tile(picked, workers, console.logged("stats"))
    store.write_stats_file(aggregate.dataset_stats(measured))
    drawn = {one.tile for one in compute_labels(force) if one.drawn}
    store.write_stats_file(
        aggregate.dataset_stats([one for one in measured if one.window.tile in drawn]),
        paths.EVALUATION_STATS_ROOT,
    )


@handler(outputs=[_COVERAGE, _SUMMARY, _SELECTION, _STATS, _LABELS])
def run_pipeline(project, force: bool = False, workers: int | None = None):
    """Run every stage on DigitalHub and publish everything each one left on disk.

    Args:
        project: The DigitalHub project the archives are logged into.
        force: Whether to redo finished work rather than skip it.
        workers: How many jobs each stage runs at once, as the job was sized.

    Returns:
        coverage: The archive of the coverage events and summaries.
        summary: The table of one row per tile and instrument set.
        selection: The archive of the tiles and observations kept.
        stats: The archive of what the filter left of the dataset.
        labels: The archive of every labelled tile.

    Raises:
        RuntimeError: When the measuring stage reported a failure, which leaves
            the coverage too incomplete to select a dataset from.
    """
    os.environ[PLAIN_LOG_ENV] = "1"
    print("measuring coverage", flush=True)
    failed = compute_coverage(force, workers)
    coverage = archived(project, _COVERAGE)
    print("uploading the summary", flush=True)
    summary = project.log_artifact(
        name=_SUMMARY,
        kind="artifact",
        source=str(paths.COVERAGE_ROOT / paths.SUMMARY_NAME),
        path=archives.published_at(project, archives.ANALYSIS_DIR, paths.SUMMARY_NAME),
        description="One row per tile and instrument set.",
    )
    archived(project, _METADATA)
    # Report a failure only once uploaded, and never select from short coverage
    if failed:
        raise RuntimeError("the run had failures; the archives hold what finished")
    compute_selection(workers, force)
    print("done", flush=True)
    return (
        coverage,
        summary,
        *(archived(project, name) for name in (_SELECTION, _STATS, _LABELS)),
    )


@handler(outputs=[_SELECTION, _STATS, _LABELS])
def run_selection(project, force: bool = False, workers: int | None = None):
    """Select the dataset on DigitalHub under the filter, and publish what it leaves.

    Args:
        project: The DigitalHub project the archives are logged into.
        force: Whether to fetch the feature catalogue again.
        workers: How many processes to run on at once, as the job was sized.

    Returns:
        selection: The archive of the tiles and observations kept.
        stats: The archive of what the filter left of the dataset.
        labels: The archive of every labelled tile.
    """
    os.environ[PLAIN_LOG_ENV] = "1"
    print("fetching the measurements", flush=True)
    archives.unpack_archive(project, _COVERAGE, paths.COVERAGE_ROOT)
    compute_selection(workers, force)
    print("done", flush=True)
    return tuple(archived(project, name) for name in (_SELECTION, _STATS, _LABELS))


def main() -> int:
    """Run the pipeline where it was asked for, over the stages it was asked for.

    Returns:
        code: A process exit code, non zero when a stage failed or an image did not
            build.
    """
    parsed = argparse.ArgumentParser(description=__doc__)
    parsed.add_argument(
        "--dh", action="store_true", help="submit to DigitalHub instead of running here"
    )
    parsed.add_argument(
        "--only-stats",
        action="store_true",
        help="skip the download and the measurement, and select, read the stats "
        "and label alone",
    )
    parsed.add_argument(
        "--force", action="store_true", help="redo finished work rather than skip it"
    )
    parsed.add_argument("--ref", default="main", help="branch, tag, or commit to run")
    arguments = parsed.parse_args()

    if arguments.dh:
        if arguments.only_stats:
            return submit.submitted(
                "selection", SELECTION_HANDLER, arguments.ref, force=arguments.force
            )
        return submit.submitted(
            "pipeline", PIPELINE_HANDLER, arguments.ref, force=arguments.force
        )
    failed = 0 if arguments.only_stats else compute_coverage(arguments.force)
    compute_selection(force=arguments.force)
    return failed


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print_interrupted("finished files")
        raise SystemExit(130) from None
