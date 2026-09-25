"""The coverage job: one instrument set measured on every tile of its group."""

from __future__ import annotations

from analysis.coverage.artifacts import write_coverage
from analysis.coverage.measurement import measure
from analysis.coverage.projection import project
from analysis.metadata.loaders.observations import load_observations
from analysis.models.job import CoverageJob, Outcome


def compute_coverage(job: CoverageJob, grid_cells: int, union_threads: int) -> Outcome:
    """Measure one instrument set's coverage of every tile of its group, and write it.

    Args:
        job: The instrument set being computed, naming what it reads and writes.
        grid_cells: How many cells one stretch of a tile's grid holds per axis.
        union_threads: How many of a tile's cells to accumulate at once.

    Returns:
        outcome: The outcome, carrying the error when the job failed.
    """
    try:
        projected, discarded = project.project_every_tile(
            load_observations(job.source), job.group.tiles
        )
        measured = [
            measure.measure_set(projected_set, grid_cells, union_threads)
            for projected_set in projected
        ]
        events = [event for set_events, _ in measured for event in set_events]
        write_coverage(job, events, [summary for _, summary in measured])
        return Outcome(job=job, events=len(events), discarded=discarded)
    except Exception as exc:
        return Outcome(job=job, error=exc)
