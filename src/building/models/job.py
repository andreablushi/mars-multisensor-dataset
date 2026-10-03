"""The unit of work a build runs, and what it reports back."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from building.metadata.observation import ObservationMetadata
from building.metadata.tile import TileMetadata
from common.models.tile import Tile


@dataclass(frozen=True, slots=True)
class Job:
    """One product to bring down and cut to every tile that kept it.

    Attributes:
        instrument: The instrument that fetches it, as ODE names it.
        identifier: What that instrument is asked for, its observation or sheet.
        frames: The tiles to cut it to, each with its own local frame.
        t_start: When the product was taken, or None.
    """

    instrument: str
    identifier: str
    frames: tuple[Tile, ...]
    t_start: datetime | None

    @property
    def label(self) -> str:
        """Return what was asked for, and the instrument it was asked of."""
        return f"{self.identifier} [{self.instrument}]"


@dataclass(frozen=True, slots=True)
class Outcome:
    """The result of running one job.

    Attributes:
        job: The job that was run.
        records: What each crop it wrote is, for the index to be built from.
        emptied: The tiles its crop reached or measured none of, dropped whole.
        error: The error raised, or None on success.
    """

    job: Job
    records: tuple[ObservationMetadata, ...] = ()
    emptied: tuple[str, ...] = ()
    error: Exception | None = None

    @property
    def failed(self) -> bool:
        """Return whether the job raised an error."""
        return self.error is not None


def lacking_tiles(outcomes: Iterable[Outcome]) -> dict[str, set[str]]:
    """Return the tiles a failed job left without its crop, by instrument.

    Args:
        outcomes: What every job of the build left.

    Returns:
        lacking: The names of the tiles each instrument left incomplete.
    """
    lacking: dict[str, set[str]] = defaultdict(set)
    for one in outcomes:
        if not one.failed:
            continue
        covered = {record.tile for record in one.records} | set(one.emptied)
        lacking[one.job.instrument].update(
            frame.name for frame in one.job.frames if frame.name not in covered
        )
    return lacking


@dataclass(frozen=True, slots=True)
class Plan:
    """The work one build has to do.

    Attributes:
        jobs: The products that still need building.
        tiles: What the dataset holds about every tile the build covers.
        skipped: Crops already written, which were left out of the jobs.
        unread: Kept observations no instrument here could read, never planned.
    """

    jobs: tuple[Job, ...]
    tiles: tuple[TileMetadata, ...]
    skipped: int
    unread: int
