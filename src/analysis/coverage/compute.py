"""Computing one instrument set's coverage of every tile of a group, start to finish."""

from __future__ import annotations

from analysis.coverage.artifacts import write
from analysis.coverage.measurement import measure
from analysis.coverage.projection import project
from analysis.metadata.loaders.observations import load_observations
from analysis.models.job import Job, Outcome


def compute(job: Job, grid_cells: int, union_threads: int) -> Outcome:
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
            measure.measure_set(one, grid_cells, union_threads) for one in projected
        ]
        events = [event for held, _ in measured for event in held]
        write.write_coverage(job, events, [summary for _, summary in measured])
        return Outcome(job=job, events=len(events), discarded=discarded)
    except Exception as exc:
        return Outcome(job=job, error=exc)
