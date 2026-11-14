"""The unit of work a build runs, and what it reports back."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import datetime

from mars_multisensor_dataset.building.metadata.observation import ObservationMetadata
from mars_multisensor_dataset.building.metadata.tile import TileMetadata
from mars_multisensor_dataset.common.models.tile import Tile


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
        emptied: The tiles its crop measured none of or held no whole patch of.
        discarded: The tiles that lost its crop but keep their others.
        error: The error raised, or None on success.
    """

    job: Job
    records: tuple[ObservationMetadata, ...] = ()
    emptied: tuple[str, ...] = ()
    discarded: tuple[str, ...] = ()
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
        covered = (
            {record.tile for record in one.records}
            | set(one.emptied)
            | set(one.discarded)
        )
        lacking[one.job.instrument].update(
            frame.name for frame in one.job.frames if frame.name not in covered
        )
    return lacking


def settled_outcomes(
    outcomes: Iterable[Outcome], held: Iterable[tuple[str, str]]
) -> list[Outcome]:
    """Return the outcomes, a discarded tile with no crop of its instrument emptied.

    Args:
        outcomes: What every job of the build left.
        held: The tile and instrument of every crop earlier runs left in the index.

    Returns:
        outcomes: The same outcomes, a tile no crop of the instrument holds emptied.
    """
    outcomes = list(outcomes)
    kept = set(held) | {
        (record.tile, one.job.instrument) for one in outcomes for record in one.records
    }
    return [
        replace(
            one,
            emptied=one.emptied
            + tuple(
                name for name in one.discarded if (name, one.job.instrument) not in kept
            ),
            discarded=tuple(
                name for name in one.discarded if (name, one.job.instrument) in kept
            ),
        )
        for one in outcomes
    ]


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
