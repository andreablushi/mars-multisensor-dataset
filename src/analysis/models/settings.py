"""What one run was asked to do, once every source has been read."""

from __future__ import annotations

from dataclasses import dataclass

from analysis.models.instrument import InstrumentSet


@dataclass(frozen=True, slots=True)
class Settings:
    """The settled choices for a run, read from one flat config file.

    Attributes:
        tile_km: The side every tile of the grid is sized to, in kilometres.
        tile_group_deg: The side every group of tiles is sized to, in degrees.
        grid_cells: Cells along each axis of every 100 km of a tile.
        instrument_sets: The instrument sets to download for every group, which
        the figures draw in that order.
        loc: "f" for every footprint overlapping the box, "o" for only those inside.
        workers: How many jobs each half runs at once.
        union_threads: How many threads one coverage job accumulates on, which is
        the share of the machine one worker gets rather than a setting of its own.
    """

    tile_km: float
    tile_group_deg: float
    grid_cells: int
    instrument_sets: tuple[InstrumentSet, ...]
    loc: str
    workers: int
    union_threads: int
