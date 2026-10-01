"""Which observations are a lit look at the tile rather than a clip of its edge."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from analysis.coverage.models.coverage import Event, SetCoverage
from analysis.metadata.loaders.observations import read_incidences
from analysis.selector.models.search_grid import SearchGrid
from analysis.selector.models.track import Offered
from analysis.utils import mask as packing
from analysis.utils.tile_group import group_of_tile_named


def admitted_observations(
    coverage: Sequence[SetCoverage],
    grid: SearchGrid,
    min_pixels: Sequence[float],
    solar_zenith: Mapping[str, float],
) -> tuple[Offered, Offered]:
    """Keep every lit observation big enough for the tile, and turn the rest away.

    Args:
        coverage: The tile's instrument sets, in any order.
        grid: The grid the tile is searched over.
        min_pixels: The pixels each set has to land on the tile, by set.
        solar_zenith: The solar zenith angle past which a look is refused, by iid.

    Returns:
        admitted: What the tile keeps, with each set and the cells it fills.
        refused: What it turned away, carrying the same.
    """
    admitted: Offered = []
    refused: Offered = []
    incidences = read_incidences(group_of_tile_named(coverage[0].summary.tile))
    for owner, instrument in enumerate(coverage):
        limit = solar_zenith.get(instrument.summary.iid, float("inf"))
        for observation in instrument.events:
            cells = [
                cell
                for cell in packing.filled_cells(observation.mask).tolist()
                if cell in grid.inside
            ]
            if not cells:
                continue
            landed = landed_pixels(observation, len(cells), grid.cell_km2)
            lit = incidences.get(observation.pdsid, 0.0) <= limit
            verdict = admitted if lit and landed >= min_pixels[owner] else refused
            verdict.append((observation, owner, cells))
    return admitted, refused


def landed_pixels(observation: Event, cells: int, cell_km2: float) -> float:
    """Return how many pixels one observation landed inside the tile.

    Args:
        observation: The observation, carrying what it covered and what it landed.
        cells: How many of the tile's own cells its footprint fills.
        cell_km2: How much ground one of those cells covers.

    Returns:
        pixels: Its pixels, scaled to the part of its footprint the tile holds.
    """
    if not observation.own_km2:
        return 0.0
    return observation.pixels * cells * cell_km2 / observation.own_km2
