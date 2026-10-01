"""What one run was asked to do, once its config has been read."""

from __future__ import annotations

from dataclasses import dataclass

from analysis.ground_truth.models.settings import GroundTruthSettings
from analysis.models.instrument import InstrumentSet
from analysis.selector.models.criteria import Criteria


@dataclass(slots=True)
class AnalysisSettings:
    """The settled choices for a run, read from the analysis config.

    Attributes:
        tile_km: The side every tile of the grid is sized to, in kilometres.
        tile_group_deg: The side every group of tiles is sized to, in degrees.
        grid_cells: Cells along each axis of every 100 km of a tile.
        instruments: The instrument sets to download for every group, by key.
        loc: Which products a box returns: b box overlap, f footprint overlap,
            o inside, i containing.
        workers: How many jobs each half runs at once.
        criteria: What a window has to hold before a tile earns a place.
        ground_truth: How the tiles the selection kept are labelled.
        sharad_distortion: The column of SHARAD's geometry table holding each row's
            signal phase distortion.
    """

    tile_km: float
    tile_group_deg: float
    grid_cells: int
    instruments: list[str]
    loc: str
    workers: int
    criteria: Criteria
    ground_truth: GroundTruthSettings
    sharad_distortion: str

    @property
    def instrument_sets(self) -> list[InstrumentSet]:
        """Return the instrument sets the keys name, in the order the config does."""
        return [InstrumentSet.from_key(key) for key in self.instruments]
