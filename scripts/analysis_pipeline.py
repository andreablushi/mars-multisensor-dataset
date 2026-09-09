#!/usr/bin/env python
"""The analysis pipeline: run here by default, or submitted with --dh."""

from __future__ import annotations

import argparse
import os
import time

from dhub import archives, submit
from dhub import configs as platform
from digitalhub_runtime_python import handler
from rich.console import Console

from analysis import configs, console, paths, planner, runner
from analysis.coverage.artifacts import index
from analysis.metadata import file_explorer
from analysis.models.progress import CoverageSummary, DownloadSummary
from analysis.selector import select
from analysis.stats.artifacts import store
from analysis.stats.dataset import aggregate, read
from shared.console import PLAIN_LOG_ENV, print_interrupted

PIPELINE_HANDLER = "scripts.analysis_pipeline:run_pipeline"
SELECTION_HANDLER = "scripts.analysis_pipeline:run_selection"

_PUBLISHED = platform.load().publishes
_COVERAGE = _PUBLISHED["coverage"]
_CATALOG = _PUBLISHED["catalog"]
_METADATA = _PUBLISHED["metadata"]
_SELECTION = _PUBLISHED["selection"]
_STATS = _PUBLISHED["stats"]
_SUMMARY = _PUBLISHED["summary"]

# Where each archive is packed from and what it holds, said once since two
# handlers publish the same ones.
ARCHIVED = {
    _COVERAGE: (
        paths.COVERAGE_ROOT,
        "Coverage events and summaries; unpack under data/analysis/.",
    ),
    _CATALOG: (
        paths.CATALOG_ROOT,
        "The ODE feature and instrument sets; unpack under data/.",
    ),
    _METADATA: (
        paths.METADATA_ROOT,
        "The ODE records behind each measurement; unpack under data/analysis/.",
    ),
    _SELECTION: (
        paths.SELECTION_ROOT,
        "The features and observations the filter keeps; unpack under data/analysis/.",
    ),
    _STATS: (
        paths.STATS_ROOT,
        "What the filter left of the dataset; unpack under data/analysis/.",
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


def compute_selection(workers: int | None = None) -> None:
    """Search every measured feature under the filter and read what it left.

    Args:
        workers: How many processes to run on at once, or None for the config.
    """
    workers = configs.load(workers=workers).workers
    picked = select.select_dataset(workers, console.logged("selection"))
    kept = sum(1 for one in picked if one.feature.kept)
    print(f"{kept:,} of {len(picked):,} features earned a place", flush=True)
    # Read off the selection just written, so they never stand for an old filter
    store.write_stats_file(
        aggregate.dataset_stats(
            read.measure_every_feature(picked, workers, console.logged("stats"))
        )
    )


@handler(outputs=[_COVERAGE, _SUMMARY, _SELECTION, _STATS])
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
        description="One row per feature and instrument set.",
    )
    archived(project, _CATALOG)
    archived(project, _METADATA)
    # Report a failure only once uploaded, and never select from short coverage
    if failed:
        raise RuntimeError("the run had failures; the archives hold what finished")
    compute_selection(workers)
    print("done", flush=True)
    return coverage, summary, archived(project, _SELECTION), archived(project, _STATS)


@handler(outputs=[_SELECTION, _STATS])
def run_selection(project, workers: int | None = None):
    """Select the dataset on DigitalHub under the filter, and publish what it leaves.

    Args:
        project: The DigitalHub project the archives are logged into.
        workers: How many processes to run on at once, as the job was sized.

    Returns:
        selection: The archive of the features and observations kept.
        stats: The archive of what the filter left of the dataset.
    """
    os.environ[PLAIN_LOG_ENV] = "1"
    print("fetching the measurements", flush=True)
    measured = project.get_artifact(_COVERAGE).download(overwrite=True)
    archives.unpack_archive(measured, paths.COVERAGE_ROOT)
    # The selection writes each feature's own ground, which it reads here
    catalogued = project.get_artifact(_CATALOG).download(overwrite=True)
    archives.unpack_archive(catalogued, paths.CATALOG_ROOT)
    compute_selection(workers)
    print("done", flush=True)
    return archived(project, _SELECTION), archived(project, _STATS)


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
        print_interrupted("finished files")
        raise SystemExit(130) from None
