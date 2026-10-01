#!/usr/bin/env python
"""The analysis pipeline: run here by default, or submitted with --dh."""

from __future__ import annotations

import os
import time
from collections import Counter

from dhub import archives, args, submit
from dhub.paths import Artifact, Function
from digitalhub_runtime_python import handler
from rich.console import Console

from analysis import paths, planner, runner
from analysis.console import print_summary
from analysis.coverage import artifacts as coverage_artifacts
from analysis.ground_truth import artifacts, catalogue
from analysis.ground_truth.candidates import candidate_labels
from analysis.ground_truth.draw import drawn_labels
from analysis.ground_truth.models.label import Label
from analysis.metadata.fetchers.ancillary import write_distortions
from analysis.selector import select
from analysis.selector.artifacts import read_selected_tiles
from analysis.stats.artifacts import write_stats
from analysis.stats.dataset import dataset_stats
from common.config import analysis_settings
from common.console import PLAIN_LOG_ENV

SELECTION_ARTIFACTS = (Artifact.SELECTION, Artifact.STATS, Artifact.LABELS)


def compute_coverage(force: bool = False, workers: int | None = None) -> int:
    """Download the ODE metadata still missing, measure it, and read its ancillary.

    Args:
        force: Whether to redo finished work rather than skip it.
        workers: How many jobs each half runs at once, or None for the config.

    Returns:
        code: A process exit code, non zero when either half had a failure.
    """
    settings = analysis_settings(workers)
    console = Console()
    started_at = time.monotonic()
    downloaded, measured = runner.pipeline_outcomes(settings, console, force)
    elapsed = time.monotonic() - started_at
    coverage_artifacts.write_index()
    print_summary(downloaded, measured, elapsed, planner.unmeasured_sources(), console)
    unread = write_distortions(settings, force)
    console.print(f"ancillary: {unread} tables left unread")
    failed = any(outcome.failed for outcome in [*downloaded, *measured])
    return 1 if failed or unread else 0


def compute_labels(force: bool) -> list[Label]:
    """Label every kept tile, draw the balanced set held out of training, and write it.

    Args:
        force: Whether to fetch the feature catalogue again rather than read it.

    Returns:
        labels: Every labelled tile, the drawn ones marked so.
    """
    settings = analysis_settings().ground_truth
    refused = catalogue.read_refused()
    labels = drawn_labels(
        candidate_labels(
            read_selected_tiles(),
            catalogue.read_features(refresh=force),
            settings,
        ),
        settings,
        refused,
    )
    artifacts.write_labels(labels)
    drawable = Counter(
        labelled.label for labelled in labels if labelled.tile not in refused
    )
    drawn = Counter(labelled.label for labelled in labels if labelled.drawn)
    for name in settings.classes:
        print(f"labels: {drawn[name]} of {drawable[name]} {name} tiles drawn")
    return labels


def compute_selection(force: bool = False, workers: int | None = None) -> None:
    """Search every measured tile under the filter, label it, and read what both left.

    Args:
        force: Whether to fetch the feature catalogue again rather than read it.
        workers: How many processes to run on at once, or None for the config.
    """
    workers = analysis_settings(workers).workers
    # Read off the selection just written, so they never stand for an old filter
    selection, measured = select.select_dataset(workers)
    kept = sum(1 for selected in selection if selected.tile.kept)
    print(f"selection: {kept:,} of {len(selection):,} tiles kept", flush=True)
    write_stats(dataset_stats(measured, selection))
    drawn = {labelled.tile for labelled in compute_labels(force) if labelled.drawn}
    write_stats(
        dataset_stats(
            [tile_stats for tile_stats in measured if tile_stats.window.tile in drawn],
            selection,
        ),
        paths.EVALUATION_STATS_ROOT,
    )


@handler(
    outputs=[
        artifact.published
        for artifact in (Artifact.COVERAGE, Artifact.METADATA, *SELECTION_ARTIFACTS)
    ]
)
def run_pipeline(project, force: bool = False, workers: int | None = None):
    """Run every stage on DigitalHub and publish everything each one left on disk.

    Args:
        project: The DigitalHub project the archives are logged into.
        force: Whether to redo finished work rather than skip it.
        workers: How many jobs each stage runs at once, as the job was sized.

    Returns:
        coverage: The archive of the coverage events and summaries.
        metadata: The archive of the ODE records the coverage was measured from.
        selection: The archive of the tiles and observations kept.
        stats: The archive of what the filter left of the dataset.
        labels: The archive of every labelled tile.

    Raises:
        RuntimeError: When the measuring stage reported a failure.
    """
    os.environ[PLAIN_LOG_ENV] = "1"
    print("measuring coverage", flush=True)
    failed = compute_coverage(force, workers)
    coverage = archives.published_artifact(project, Artifact.COVERAGE)
    metadata = archives.published_artifact(project, Artifact.METADATA)
    # Report a failure only once uploaded, and never select from short coverage
    if failed:
        raise RuntimeError("the run had failures; the archives hold what finished")
    archives.download_artifact(project, Artifact.VERDICTS)
    compute_selection(force, workers)
    print("done", flush=True)
    return (
        coverage,
        metadata,
        *(
            archives.published_artifact(project, artifact)
            for artifact in SELECTION_ARTIFACTS
        ),
    )


@handler(outputs=[artifact.published for artifact in SELECTION_ARTIFACTS])
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
    archives.download_artifact(project, Artifact.COVERAGE)
    archives.download_artifact(project, Artifact.METADATA)
    archives.download_artifact(project, Artifact.VERDICTS)
    compute_selection(force, workers)
    print("done", flush=True)
    return tuple(
        archives.published_artifact(project, artifact)
        for artifact in SELECTION_ARTIFACTS
    )


def main() -> int:
    """Run the pipeline where it was asked for, over the stages it was asked for.

    Returns:
        code: A process exit code, non zero when a stage failed.
    """
    parsed = args.script_parser(__doc__, "redo finished work rather than skip it")
    parsed.add_argument(
        "--only-stats",
        action="store_true",
        help="skip the download and the measurement, and select, read the stats "
        "and label alone",
    )
    arguments = parsed.parse_args()

    if arguments.dh:
        stage = Function.SELECTION if arguments.only_stats else Function.PIPELINE
        return submit.submitted(stage, arguments.ref, force=arguments.force)
    failed = 0 if arguments.only_stats else compute_coverage(arguments.force)
    compute_selection(force=arguments.force)
    return failed


if __name__ == "__main__":
    args.run_script(main, "finished files")
