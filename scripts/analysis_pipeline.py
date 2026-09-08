#!/usr/bin/env python
"""The analysis pipeline: run here by default, or submitted with --dh."""

from __future__ import annotations

import argparse
import os
import time

from dhub import archives, configs, submit
from digitalhub_runtime_python import handler
from rich.console import Console

import analysis.utils.settings as settings
import utils.disk.paths as paths
from analysis import console, planner, runner
from analysis.coverage.artifacts import index
from analysis.metadata import file_explorer
from analysis.models.progress import CoverageSummary, DownloadSummary
from analysis.selector import select
from analysis.stats.artifacts import store
from analysis.stats.dataset import aggregate, read

PIPELINE_HANDLER = "scripts.analysis_pipeline:run_pipeline"
SELECTION_HANDLER = "scripts.analysis_pipeline:run_selection"

_PUBLISHED = configs.load().publishes
_COVERAGE = _PUBLISHED["coverage"]
_CATALOG = _PUBLISHED["catalog"]
_METADATA = _PUBLISHED["metadata"]
_SUMMARY = _PUBLISHED["summary"]

SELECTION_ARCHIVES = (
    (
        _PUBLISHED["selection"],
        paths.SELECTION_ROOT,
        "The features and observations the filter keeps; unpack under data/analysis/.",
    ),
    (
        _PUBLISHED["stats"],
        paths.STATS_ROOT,
        "What the filter left of the dataset; unpack under data/analysis/.",
    ),
)


def compute_coverage(force: bool = False, workers: int | None = None) -> int:
    """Download the ODE metadata still missing and measure the coverage it left.

    Args:
        force: Whether to redo finished work rather than skip it.
        workers: How many jobs each half runs at once, or None for the config.

    Returns:
        code: A process exit code, non zero when either half had a failure.
    """
    choices = settings.load(workers=workers)
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


def compute_selection(workers: int | None = None) -> None:
    """Search every measured feature under the filter and read what it left.

    Args:
        workers: How many processes to run on at once, or None for the config.
    """
    workers = settings.load(workers=workers).workers
    picked = select.select_dataset(workers, console.logged("selection"))
    kept = sum(1 for one in picked if one.feature.kept)
    print(f"{kept:,} of {len(picked):,} features earned a place", flush=True)
    # Read off the selection just written, so they never stand for an old filter
    store.write_stats_file(
        aggregate.dataset_stats(
            read.measure_every_feature(picked, workers, console.logged("stats"))
        )
    )


@handler(outputs=[_COVERAGE, _SUMMARY, *(name for name, _, _ in SELECTION_ARCHIVES)])
def run_pipeline(project, force: bool = False, workers: int | None = None):
    """Run every stage on DigitalHub and publish everything each one left on disk.

    Args:
        project: The DigitalHub project the archives are logged into.
        force: Whether to redo finished work rather than skip it.
        workers: How many jobs each stage runs at once, as the job was sized.

    Returns:
        coverage: The archive of the coverage events and summaries.
        summary: The table of one row per feature and instrument set.
        selection: The archive of the features and observations kept.
        stats: The archive of what the filter left of the dataset.

    Raises:
        RuntimeError: When the measuring stage reported a failure, which leaves
            the coverage too incomplete to select a dataset from.
    """
    os.environ[console.PLAIN_LOG_ENV] = "1"
    print("measuring coverage", flush=True)
    failed = compute_coverage(force, workers)
    coverage = archives.logged_archive(
        project,
        paths.COVERAGE_ROOT,
        _COVERAGE,
        "Coverage events and summaries; unpack under data/analysis/.",
    )
    print("uploading the summary", flush=True)
    summary = project.log_table(
        name=_SUMMARY,
        source=str(paths.COVERAGE_ROOT / paths.SUMMARY_NAME),
        description="One row per feature and instrument set.",
    )
    archives.logged_archive(
        project,
        paths.CATALOG_ROOT,
        _CATALOG,
        "The ODE feature and instrument sets; unpack under data/.",
    )
    archives.logged_archive(
        project,
        paths.METADATA_ROOT,
        _METADATA,
        "The ODE records behind each measurement; unpack under data/analysis/.",
    )
    # Report a failure only once uploaded, and never select from short coverage
    if failed:
        raise RuntimeError("the run had failures; the archives hold what finished")
    compute_selection(workers)
    print("done", flush=True)
    return (
        coverage,
        summary,
        *(
            archives.logged_archive(project, root, name, held)
            for name, root, held in SELECTION_ARCHIVES
        ),
    )


@handler(outputs=[name for name, _, _ in SELECTION_ARCHIVES])
def run_selection(project, workers: int | None = None):
    """Select the dataset on DigitalHub under the filter, and publish what it leaves.

    Args:
        project: The DigitalHub project the archives are logged into.
        workers: How many processes to run on at once, as the job was sized.

    Returns:
        selection: The archive of the features and observations kept.
        stats: The archive of what the filter left of the dataset.
    """
    os.environ[console.PLAIN_LOG_ENV] = "1"
    print("fetching the measurements", flush=True)
    archives.unpacked(
        project.get_artifact(_COVERAGE).download(overwrite=True), paths.COVERAGE_ROOT
    )
    # The selection writes each feature's own ground, which it reads here
    archives.unpacked(
        project.get_artifact(_CATALOG).download(overwrite=True), paths.CATALOG_ROOT
    )
    compute_selection(workers)
    print("done", flush=True)
    return tuple(
        archives.logged_archive(project, root, name, held)
        for name, root, held in SELECTION_ARCHIVES
    )


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
        help="skip the download and the measurement, and read the stats alone",
    )
    parsed.add_argument(
        "--force", action="store_true", help="redo finished work rather than skip it"
    )
    parsed.add_argument("--ref", default="main", help="branch, tag, or commit to run")
    arguments = parsed.parse_args()

    if arguments.dh:
        if arguments.only_stats:
            return submit.submitted("selection", SELECTION_HANDLER, arguments.ref)
        return submit.submitted(
            "pipeline", PIPELINE_HANDLER, arguments.ref, force=arguments.force
        )
    failed = 0 if arguments.only_stats else compute_coverage(arguments.force)
    compute_selection()
    return failed


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        console.print_interrupted()
        raise SystemExit(130) from None
