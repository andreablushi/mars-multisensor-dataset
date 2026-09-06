"""Running the pipeline's work: one pooled runner, and the two halves it drives."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import (
    Executor,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from contextlib import closing

from rich.console import Console

from analysis import planner
from analysis.console import describe, render
from analysis.coverage import compute
from analysis.metadata import download, file_explorer
from analysis.metadata.loaders.features import load_features
from analysis.metadata.ode import ODEClient
from analysis.models.job import Job, Outcome
from analysis.models.progress import ProgressEvent
from analysis.models.settings import Settings


def run_jobs(
    jobs: Sequence[Job], execute: Callable[[Job], Outcome], pool: Executor
) -> Iterator[ProgressEvent]:
    """Run every job on the pool, yielding progress as each one finishes.

    Args:
        jobs: The work to run.
        execute: What to call for one job, which must never raise and must be picklable.
        pool: The pool to run on, owned and shut down by the caller.

    Yields:
        One event per finished job, in completion order.
    """
    futures = [pool.submit(execute, job) for job in jobs]
    for completed, future in enumerate(as_completed(futures), start=1):
        yield ProgressEvent(completed=completed, outcome=future.result())


def run_pipeline(
    settings: Settings, console: Console, force: bool = False
) -> tuple[list[Outcome], list[Outcome]]:
    """Download every set still missing and measure every set not yet measured.

    Args:
        settings: The settled choices for the run.
        console: The console to render on.
        force: Whether to redo finished work rather than skip it, which covers
            both halves at once: a set is downloaded again and measured again.

    Returns:
        Every finished download outcome, then every finished coverage outcome.
    """
    futures: list[Future[Outcome]] = []
    fetched: list[Outcome] = []
    with ODEClient() as client:
        features = load_features(client, refresh=settings.refresh_catalog)
        plan = planner.download_plan(features, settings.instrument_sets, force=force)
        rewriting = {job.output_path for job in plan.jobs}
        stored = [held for held in file_explorer.find_sets() if held not in rewriting]
        backlog = planner.coverage_plan(stored, force=force)
        describe(plan, backlog, settings, console)
        with (
            ProcessPoolExecutor(max_workers=settings.workers) as measuring,
            ThreadPoolExecutor(max_workers=settings.workers) as fetching,
        ):

            def measure(job: Job) -> Future[Outcome]:
                """Put one coverage job on the pool, sized as the run is configured."""
                return measuring.submit(
                    compute.compute, job, settings.grid_cells, settings.union_threads
                )

            def measured() -> Iterator[ProgressEvent]:
                """Pass each download through, keeping it and measuring what landed."""
                downloads = run_jobs(
                    plan.jobs,
                    lambda job: download.download(job, client, settings.loc),
                    fetching,
                )
                for event in downloads:
                    fetched.append(event.outcome)
                    source = event.outcome.job.output_path
                    if not event.outcome.failed and source.stat().st_size:
                        futures.extend(
                            measure(job)
                            for job in planner.coverage_plan([source], force=force).jobs
                        )
                    yield event

            futures.extend(measure(job) for job in backlog.jobs)
            with closing(measured()) as events:
                render(events, len(plan.jobs), "download", console)
            if futures:
                render(
                    (
                        ProgressEvent(completed=done, outcome=future.result())
                        for done, future in enumerate(futures, start=1)
                    ),
                    len(futures),
                    "coverage",
                    console,
                )
    return fetched, [future.result() for future in futures]
