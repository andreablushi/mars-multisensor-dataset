"""Computing one instrument set's coverage of its feature, start to finish."""

from __future__ import annotations

from analysis.coverage.artifacts import write
from analysis.coverage.measurement import measure
from analysis.coverage.projection import project
from analysis.metadata.loaders.observations import load_observations
from analysis.models.job import Job, Outcome


def compute(job: Job, grid_cells: int, union_threads: int) -> Outcome:
    """Measure one instrument set's coverage of its feature and write it out.

    Args:
        job: The instrument set being computed, naming what it reads and writes.
        grid_cells: How many cells one block of the feature's grid holds per axis.
        union_threads: How many of the feature's cells to accumulate at once.

    Returns:
        outcome: The outcome, carrying the error when the job failed.
    """
    try:
        projected = project.project(load_observations(job.source))
        if not projected.observations:
            return Outcome(job=job, discarded=projected.discarded)
        events, summary = measure.measure_set(projected, grid_cells, union_threads)
        write.write_coverage(job, events, summary)
        return Outcome(job=job, events=len(events), discarded=projected.discarded)
    except Exception as exc:
        return Outcome(job=job, error=exc)
