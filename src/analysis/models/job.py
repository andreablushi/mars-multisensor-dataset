"""The unit of work each stage runs, and what it reports back."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from analysis.models.instrument import InstrumentSet
from analysis.models.tile_group import TileGroup


@dataclass(frozen=True, slots=True)
class DownloadJob:
    """One group and instrument set to download.

    Attributes:
        group: The group of tiles the job is run over.
        instrument_set: The instrument set to query.
        output_path: The JSONL file the results are written to.
    """

    group: TileGroup
    instrument_set: InstrumentSet
    output_path: Path

    @property
    def label(self) -> str:
        """Return the group and instrument set key, for progress lines."""
        return f"{self.group.name} [{self.instrument_set.key}]"


@dataclass(frozen=True, slots=True)
class CoverageJob:
    """One stored instrument set to compute coverage for.

    Attributes:
        group: The group of tiles the job is run over.
        source: The JSONL file holding one instrument set's observations.
        events_path: The parquet file the per-observation rows go to.
        summary_path: The parquet file the set's summary rows go to, written last.
    """

    group: TileGroup
    source: Path
    events_path: Path
    summary_path: Path

    @property
    def label(self) -> str:
        """Return the group and source file stem, for progress lines."""
        return f"{self.group.name}/{self.source.stem}"


@dataclass(frozen=True, slots=True)
class Outcome:
    """The result of running one job.

    Attributes:
        job: The job that was run.
        events: How many observation rows were written.
        discarded: How many stored records could not be measured.
        error: The error raised, or None on success.
    """

    job: DownloadJob | CoverageJob
    events: int = 0
    discarded: int = 0
    error: Exception | None = None

    @property
    def label(self) -> str:
        """Return the label of the job that was run."""
        return self.job.label

    @property
    def failed(self) -> bool:
        """Return whether the job raised an error."""
        return self.error is not None


@dataclass(frozen=True, slots=True)
class Plan:
    """The work selected for one half of a run.

    Attributes:
        jobs: The jobs that still need running.
        skipped: How many were left alone, their output already on disk.
    """

    jobs: tuple[DownloadJob | CoverageJob, ...]
    skipped: int

    @classmethod
    def of(
        cls, jobs: Sequence[DownloadJob | CoverageJob], wanted: Sequence[bool]
    ) -> Plan:
        """Keep the jobs still wanted, counting the rest as skipped.

        Args:
            jobs: Every job the run could do, in the order to run them.
            wanted: Whether each one still has to run.

        Returns:
            plan: The jobs to run, and how many were skipped.
        """
        kept = tuple(job for job, want in zip(jobs, wanted, strict=True) if want)
        return cls(jobs=kept, skipped=len(jobs) - len(kept))
