"""What one run was asked to do, once every source has been read."""

from __future__ import annotations

from dataclasses import dataclass

from analysis.models.instrument import InstrumentSet


@dataclass(frozen=True, slots=True)
class Settings:
    """The settled choices for a run, read from one flat config file.

    Attributes:
        grid_cells: Cells along each axis of one block.
        instrument_sets: The instrument sets to download for every feature, which
        the figures draw in that order.
        loc: "f" for every footprint overlapping the box, "o" for only those inside.
        refresh_catalog: Whether to re-fetch the ODE feature catalogue rather than
        read the cache.
        workers: How many jobs each half runs at once.
        union_threads: How many threads one coverage job accumulates on, which is
        the share of the machine one worker gets rather than a setting of its own.
    """

    grid_cells: int
    instrument_sets: tuple[InstrumentSet, ...]
    loc: str
    refresh_catalog: bool
    workers: int
    union_threads: int
