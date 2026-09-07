"""The unit of work a build runs, and what it reports back."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from building.metadata.feature import FeatureMetadata
from building.metadata.observation import ObservationMetadata
from building.models.feature import FeatureFrame


@dataclass(frozen=True, slots=True)
class Job:
    """One product to bring down and cut to every feature that kept it.

    The same product is often kept for several features, so it is downloaded,
    read and cleaned once and then cut once per feature.

    Attributes:
        instrument: The instrument that fetches it, as ODE names it.
        identifier: What that instrument is asked for, its observation or tile.
        frames: The features to cut it to, each with its own local frame.
        t_start: When the product was taken, or None where its archive
            publishes no time for it.
    """

    instrument: str
    identifier: str
    frames: tuple[FeatureFrame, ...] = ()
    t_start: datetime | None = None

    @property
    def label(self) -> str:
        """Return a short human readable name for this job.

        Returns:
            What was asked for, and the instrument it was asked of.
        """
        return f"{self.identifier} [{self.instrument}]"


@dataclass(frozen=True, slots=True)
class Outcome:
    """The result of running one job.

    Attributes:
        job: The job that was run.
        records: What each crop it wrote is, for the index to be built from.
        missed: How many features it reached none of, which is not a failure.
        error: The error raised, or None on success.
    """

    job: Job
    records: tuple[ObservationMetadata, ...] = ()
    missed: int = 0
    error: Exception | None = None

    @property
    def written(self) -> int:
        """Return how many crops the job wrote.

        Returns:
            How many records it left, one per crop.
        """
        return len(self.records)

    @property
    def label(self) -> str:
        """Return a short human readable name for the job that was run.

        Returns:
            The label of the underlying job.
        """
        return self.job.label

    @property
    def failed(self) -> bool:
        """Return whether the job raised an error.

        Returns:
            True when an error was recorded.
        """
        return self.error is not None


@dataclass(frozen=True, slots=True)
class Plan:
    """The work one build has to do.

    Attributes:
        jobs: The products that still need building.
        features: What the dataset holds about every feature the build covers.
        skipped_existing: Products left alone because every crop of them is
            already written.
        unread: Observations the selection kept that no instrument here could
            read, whether it builds none of that instrument or the id names no
            observation of it, and which were therefore never planned.
    """

    jobs: tuple[Job, ...]
    features: tuple[FeatureMetadata, ...] = ()
    skipped_existing: int = 0
    unread: int = 0

    @property
    def instruments(self) -> tuple[str, ...]:
        """Return which instruments the build has products of.

        Returns:
            Each instrument named once, in name order.
        """
        return tuple(sorted({job.instrument for job in self.jobs}))

    @property
    def feature_count(self) -> int:
        """Return how many features the build covers.

        Returns:
            One per feature it holds.
        """
        return len(self.features)
