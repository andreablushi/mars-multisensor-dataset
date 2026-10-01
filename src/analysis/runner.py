"""The pipeline's two halves run side by side: metadata downloads and coverage jobs."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial

import httpx
from rich.console import Console

from analysis import paths, planner
from analysis.console import LOGGED_LINES, print_plans
from analysis.coverage.compute import compute_coverage
from analysis.metadata.download import download_outcome
from analysis.models.job import Outcome
from analysis.models.settings import AnalysisSettings
from analysis.utils.tile_group import every_tile_group, tile_grid
from common.console import Tracker
from common.fetch.http import TLS_CONTEXT
from common.pool import cancellable_pool


def pipeline_outcomes(
    settings: AnalysisSettings, console: Console, force: bool, cores: int | None
) -> tuple[list[Outcome], list[Outcome]]:
    """Download every set still missing and measure every set not yet measured.

    Args:
        settings: The settled choices for the run.
        console: The console to render on.
        force: Whether to download and measure finished sets again.
        cores: The cores the run is given, or None for the machine's.

    Returns:
        downloaded: Every finished download outcome.
        measured: Every finished coverage outcome.
    """
    # The coverage jobs run side by side, so each takes a share of the machine
    union_threads = max(1, (cores or os.process_cpu_count() or 1) // settings.workers)
    groups = every_tile_group(tile_grid(), settings.tile_group_deg)
    download_plan = planner.download_plan(groups, settings.instrument_sets, force=force)
    downloading = {job.output_path for job in download_plan.jobs}
    stored = [source for source in paths.metadata_files() if source not in downloading]
    coverage_plan = planner.coverage_plan(stored, groups, force=force)
    print_plans(download_plan, coverage_plan, console)
    downloaded: list[Outcome] = []
    with (
        httpx.Client(verify=TLS_CONTEXT) as client,
        cancellable_pool(ProcessPoolExecutor(settings.workers)) as coverage_pool,
        cancellable_pool(ThreadPoolExecutor(settings.workers)) as download_pool,
    ):
        measure = partial(
            compute_coverage,
            grid_cells=settings.grid_cells,
            union_threads=union_threads,
        )
        coverage_futures = [
            coverage_pool.submit(measure, job) for job in coverage_plan.jobs
        ]
        download_futures = [
            download_pool.submit(download_outcome, job, client, settings.loc)
            for job in download_plan.jobs
        ]
        with Tracker(
            "download", len(download_futures), console, LOGGED_LINES
        ) as tracker:
            for future in as_completed(download_futures):
                outcome = future.result()
                downloaded.append(outcome)
                tracker.advance(outcome.job.label, outcome.error)
                # A set is measured as soon as it lands, beside the backlog
                source = outcome.job.output_path
                if not outcome.failed and source.stat().st_size:
                    landed = planner.coverage_plan([source], groups, force=force)
                    coverage_futures += [
                        coverage_pool.submit(measure, job) for job in landed.jobs
                    ]
        with Tracker(
            "coverage", len(coverage_futures), console, LOGGED_LINES
        ) as tracker:
            for future in as_completed(coverage_futures):
                outcome = future.result()
                tracker.advance(outcome.job.label, outcome.error)
    return downloaded, [future.result() for future in coverage_futures]
