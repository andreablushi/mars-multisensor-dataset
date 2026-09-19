"""Writing one instrument set's coverage artifacts, under a derived schema."""

from __future__ import annotations

from collections.abc import Sequence

from common.analysis.coverage.models.coverage import Event
from common.analysis.coverage.models.summary import Summary
from common.analysis.models.job import Job
from common.disk import parquet

EVENTS = parquet.schema_of(Event)
SUMMARY = parquet.schema_of(Summary)


def write_coverage(
    job: Job, events: Sequence[Event], summaries: Sequence[Summary]
) -> None:
    """Write one set's observation rows and the row describing it on each tile.

    Args:
        job: The instrument set that was computed, naming both destinations.
        events: The set's observation rows, tile by tile in chronological order.
        summaries: One row per tile the set reached, describing it as a whole.
    """
    parquet.write(events, EVENTS, job.events_path)
    parquet.write(summaries, SUMMARY, job.summary_path)
