"""Which observations are a lit look at the tile rather than a clip of its edge."""

from __future__ import annotations

from collections.abc import Sequence
from math import inf

from analysis.coverage.models.coverage import SetCoverage
from analysis.metadata.loaders.observations import read_incidences
from analysis.models.instrument import PIXEL_KM2
from analysis.selector.models.criteria import Criteria
from analysis.selector.models.search_grid import SearchGrid
from analysis.selector.models.track import Offered
from analysis.utils import mask as packing
from analysis.utils.tile_group import group_of_tile_named
from building.configs import sharad


def admitted_observations(
    coverage: Sequence[SetCoverage],
    grid: SearchGrid,
    criteria: Criteria,
) -> tuple[Offered, Offered]:
    """Keep every lit observation big enough for the tile, and turn the rest away.

    Args:
        coverage: The tile's instrument sets, in any order.
        grid: The grid the tile is searched over.
        criteria: The ground and the lighting each instrument is asked for.

    Returns:
        admitted: What the tile keeps, with each set and the cells it fills.
        refused: What it turned away, carrying the same.
    """
    admitted: Offered = []
    refused: Offered = []
    incidences = read_incidences(group_of_tile_named(coverage[0].summary.tile))
    for owner, instrument in enumerate(coverage):
        iid = instrument.summary.iid
        limit = (
            inf
            if iid == sharad.LAYOUT.instrument
            else criteria.solar_zenith.get(iid, inf)
        )
        floor = floor_km2(criteria, iid)
        for observation in instrument.events:
            cells = [
                cell
                for cell in packing.filled_cells(observation.mask).tolist()
                if cell in grid.inside
            ]
            if not cells:
                continue
            landed_km2 = len(cells) * grid.cell_km2
            lit = incidences.get(observation.pdsid, inf) <= limit
            verdict = admitted if lit and landed_km2 >= floor else refused
            verdict.append((observation, owner, cells))
    return admitted, refused


def floor_km2(criteria: Criteria, iid: str) -> float:
    """Return the ground one instrument's look has to land on a tile to count."""
    return criteria.admits.get(iid, 0.0) * PIXEL_KM2[iid]
