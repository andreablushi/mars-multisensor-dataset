"""The unit of work each stage runs, and what it reports back."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from analysis.models.instrument import InstrumentSet
from shared.models.feature import Feature


@dataclass(frozen=True, slots=True)
class Job:
    """One feature and instrument set to download, or to compute coverage for.

    Attributes:
        feature: The feature to query, on a download job.
        instrument_set: The instrument set to query, on a download job.
        output_path: The JSONL file the results are written to.
        source: The JSONL file holding one instrument set's observations.
        events_path: The parquet file the per-observation rows go to.
        summary_path: The parquet file the set's one summary row goes to, written last.
    """

    feature: Feature | None = None
    instrument_set: InstrumentSet | None = None
    output_path: Path | None = None
    source: Path | None = None
    events_path: Path | None = None
    summary_path: Path | None = None

    @property
    def label(self) -> str:
        """Return a short human readable name for this job.

        Returns:
            label: The feature and instrument set, named as the job's own stage spells
                them.
        """
        if self.feature is not None and self.instrument_set is not None:
            return f"{self.feature.name} [{self.instrument_set.key}]"
        return f"{self.source.parent.name}/{self.source.stem}"


@dataclass(frozen=True, slots=True)
class Outcome:
    """The result of running one job.

    Attributes:
        job: The job that was run.
        events: How many observation rows were written.
        discarded: How many stored records could not be measured.
        error: The error raised, or None on success.
    """

    job: Job
    events: int = 0
    discarded: int = 0
    error: Exception | None = None

    @property
    def label(self) -> str:
        """Return a short human readable name for the job that was run.

        Returns:
            label: The label of the underlying job.
        """
        return self.job.label

    @property
    def failed(self) -> bool:
        """Return whether the job raised an error.

        Returns:
            failed: True when an error was recorded.
        """
        return self.error is not None

    @property
    def empty(self) -> bool:
        """Return whether the job finished having measured nothing.

        Returns:
            empty: True when the set produced no observation rows.
        """
        return self.error is None and self.events == 0


@dataclass(frozen=True, slots=True)
class Plan:
    """The work selected for one half of a run.

    Attributes:
        jobs: The jobs that still need running.
        feature_count: Features selected, or discovered on disk.
        set_count: Instrument sets selected, or discovered on disk.
        skipped_existing: Outputs left in place because they already exist.
    """

    jobs: tuple[Job, ...]
    feature_count: int
    set_count: int
    skipped_existing: int
