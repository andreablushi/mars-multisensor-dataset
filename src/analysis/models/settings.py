"""What one run was asked to do, once its config has been read."""

from __future__ import annotations

from dataclasses import dataclass, field

from analysis.ground_truth.models.settings import Settings as GroundTruth
from analysis.models.ancillary import Ancillary
from analysis.models.instrument import InstrumentSet
from analysis.selector.models.filter import Filter


@dataclass(slots=True)
class Settings:
    """The settled choices for a run, read from the analysis config.

    Attributes:
        tile_km: The side every tile of the grid is sized to, in kilometres.
        tile_group_deg: The side every group of tiles is sized to, in degrees.
        grid_cells: Cells along each axis of every 100 km of a tile.
        instruments: The instrument sets to download for every group, by key.
        loc: "f" for every footprint overlapping the box, "o" for only those inside.
        workers: How many jobs each half runs at once.
        window: What a window has to hold before a tile earns a place.
        ground_truth: How the tiles the selection kept are labelled.
        ancillary: The table published beside a set's products, by the set's key.
        union_threads: How many threads one coverage job accumulates on.
    """

    tile_km: float
    tile_group_deg: float
    grid_cells: int
    instruments: list[str]
    loc: str
    workers: int
    window: Filter
    ground_truth: GroundTruth
    ancillary: dict[str, Ancillary] = field(default_factory=dict)
    union_threads: int = 1

    @property
    def instrument_sets(self) -> list[InstrumentSet]:
        """Return the instrument sets the keys name.

        Returns:
            sets: One set per key, in the order the config names them.
        """
        return [InstrumentSet.from_key(key) for key in self.instruments]
