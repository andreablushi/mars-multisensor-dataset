"""The unit of work a build runs, and what it reports back."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from building.metadata.observation import ObservationMetadata
from building.metadata.tile import TileMetadata
from shared.models.tile import Tile


@dataclass(frozen=True, slots=True)
class Job:
    """One product to bring down and cut to every tile that kept it.

    Attributes:
        instrument: The instrument that fetches it, as ODE names it.
        identifier: What that instrument is asked for, its observation or sheet.
        frames: The tiles to cut it to, each with its own local frame.
        t_start: When the product was taken, or None where its archive
            publishes no time for it.
    """

    instrument: str
    identifier: str
    frames: tuple[Tile, ...] = ()
    t_start: datetime | None = None

    @property
    def label(self) -> str:
        """Return a short human readable name for this job.

        Returns:
            label: What was asked for, and the instrument it was asked of.
        """
        return f"{self.identifier} [{self.instrument}]"


@dataclass(frozen=True, slots=True)
class Outcome:
    """The result of running one job.

    Attributes:
        job: The job that was run.
        records: What each crop it wrote is, for the index to be built from.
        missed: How many tiles it reached none of, which is not a failure.
        error: The error raised, or None on success.
    """

    job: Job
    records: tuple[ObservationMetadata, ...] = ()
    missed: int = 0
    error: Exception | None = None


@dataclass(frozen=True, slots=True)
class Plan:
    """The work one build has to do.

    Attributes:
        jobs: The products that still need building.
        tiles: What the dataset holds about every tile the build covers.
        skipped_existing: Products left alone because every crop of them is
            already written.
        unread: Observations the selection kept that no instrument here could read,
            and which were therefore never planned.
    """

    jobs: tuple[Job, ...]
    tiles: tuple[TileMetadata, ...] = ()
    skipped_existing: int = 0
    unread: int = 0
