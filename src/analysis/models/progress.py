"""Progress reporting shared by both pipelines."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from analysis.models.job import Outcome


@dataclass(frozen=True)
class ProgressEvent:
    """A snapshot of run progress, emitted as each unit of work finishes.

    Attributes:
        completed: Units finished so far, successful or not.
        outcome: The unit that just finished.
    """

    completed: int
    outcome: Outcome


@dataclass(frozen=True, slots=True)
class DownloadSummary:
    """Totals for a finished download run.

    Attributes:
        ran: Jobs executed successfully.
        failed: Jobs that raised an error.
        elapsed: Total run duration in seconds.
    """

    ran: int
    failed: int
    elapsed: float

    @classmethod
    def from_outcomes(
        cls, outcomes: Sequence[Outcome], elapsed: float
    ) -> DownloadSummary:
        """Total up what the download half of the run did.

        Args:
            outcomes: Every finished download job.
            elapsed: How long the half took in seconds.

        Returns:
            summary: The summary.
        """
        return cls(
            ran=sum(1 for outcome in outcomes if not outcome.failed),
            failed=sum(1 for outcome in outcomes if outcome.failed),
            elapsed=elapsed,
        )


@dataclass(frozen=True, slots=True)
class CoverageSummary:
    """Totals for a finished coverage run.

    Attributes:
        computed: Instrument sets that produced rows.
        empty: Instrument sets that ran but had nothing left to measure.
        failed: Instrument sets that raised an error.
        events: Observation rows written across every set.
        discarded: Records carrying no footprint, no start time, or no overlap.
        elapsed: Total run duration in seconds.
    """

    computed: int
    empty: int
    failed: int
    events: int
    discarded: int
    elapsed: float

    @classmethod
    def from_outcomes(
        cls, outcomes: Sequence[Outcome], elapsed: float
    ) -> CoverageSummary:
        """Total up what the coverage half of the run did.

        Args:
            outcomes: Every finished coverage job.
            elapsed: How long the run took in seconds.

        Returns:
            summary: The summary.
        """
        return cls(
            computed=sum(1 for o in outcomes if not o.failed and not o.empty),
            empty=sum(1 for o in outcomes if not o.failed and o.empty),
            failed=sum(1 for o in outcomes if o.failed),
            events=sum(o.events for o in outcomes if not o.failed),
            discarded=sum(o.discarded for o in outcomes),
            elapsed=elapsed,
        )
