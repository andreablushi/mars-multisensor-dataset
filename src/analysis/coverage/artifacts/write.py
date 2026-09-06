"""Writing one instrument set's coverage artifacts, under a derived schema."""

from __future__ import annotations

from collections.abc import Sequence

from analysis.coverage.models.coverage import Event
from analysis.coverage.models.summary import Summary
from analysis.models.job import Job
from utils.disk import parquet

EVENTS = parquet.schema_of(Event)
SUMMARY = parquet.schema_of(Summary)


def write_coverage(job: Job, events: Sequence[Event], summary: Summary) -> None:
    """Write one set's observation rows and the single row describing it.

    Args:
        job: The instrument set that was computed, naming both destinations.
        events: The set's observation rows, in chronological order.
        summary: The one row describing the set as a whole.

    Returns:
        None.
    """
    parquet.write(events, EVENTS, job.events_path)
    parquet.write([summary], SUMMARY, job.summary_path)
