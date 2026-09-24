"""Running the pipeline's two halves: downloading metadata, and measuring it."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial

from rich.console import Console

from analysis import planner
from analysis.console import Tracker, describe
from analysis.coverage import compute
from analysis.metadata import download, file_explorer
from analysis.models.job import Outcome
from analysis.models.settings import Settings
from analysis.utils.tile_group import every_tile_group, tile_grid
from common.fetch.ode import ODEClient


def run_pipeline(
    settings: Settings, console: Console, force: bool = False, cores: int | None = None
) -> tuple[list[Outcome], list[Outcome]]:
    """Download every set still missing and measure every set not yet measured.

    Args:
        settings: The settled choices for the run.
        console: The console to render on.
        force: Whether to download and measure finished sets again.
        cores: The cores the run is given, or None for the machine's.

    Returns:
        fetched: Every finished download outcome.
        measured: Every finished coverage outcome.
    """
    # The coverage jobs run side by side, so each takes a share of the machine
    union_threads = max(1, (cores or os.process_cpu_count() or 1) // settings.workers)
    groups = every_tile_group(tile_grid(), settings.tile_group_deg)
    downloads = planner.download_plan(groups, settings.instrument_sets, force=force)
    rewriting = {job.output_path for job in downloads.jobs}
    stored = [source for source in file_explorer.find_sets() if source not in rewriting]
    backlog = planner.coverage_plan(stored, groups, force=force)
    describe(downloads, backlog, console)
    fetched: list[Outcome] = []
    with (
        ODEClient() as client,
        ProcessPoolExecutor(max_workers=settings.workers) as coverage_pool,
        ThreadPoolExecutor(max_workers=settings.workers) as download_pool,
    ):
        measure = partial(
            compute.compute, grid_cells=settings.grid_cells, union_threads=union_threads
        )
        coverage_futures = [coverage_pool.submit(measure, job) for job in backlog.jobs]
        download_futures = [
            download_pool.submit(download.download, job, client, settings.loc)
            for job in downloads.jobs
        ]
        with Tracker("download", len(download_futures), console) as tracker:
            for future in as_completed(download_futures):
                outcome = future.result()
                fetched.append(outcome)
                tracker.advance(outcome)
                # A set is measured as soon as it lands, beside the backlog
                source = outcome.job.output_path
                if not outcome.failed and source.stat().st_size:
                    landed = planner.coverage_plan([source], groups, force=force)
                    coverage_futures += [
                        coverage_pool.submit(measure, job) for job in landed.jobs
                    ]
        with Tracker("coverage", len(coverage_futures), console) as tracker:
            for future in coverage_futures:
                tracker.advance(future.result())
    return fetched, [future.result() for future in coverage_futures]
