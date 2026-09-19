"""Which observations are a look at the tile rather than a clip of its edge."""

from __future__ import annotations

from collections.abc import Sequence

from common.analysis.coverage.models.coverage import Event, SetCoverage
from common.analysis.selector.models.filter import Filter
from common.analysis.selector.models.grid import Grid
from common.analysis.utils import mask as packing

# The admitted observations, the set each belongs to, and the cells each fills
Held = list[tuple[Event, int, list[int]]]


def admit_observation(
    coverage: Sequence[SetCoverage], grid: Grid, criteria: Filter
) -> tuple[Held, Held]:
    """Keep every observation big enough for the tile, and turn the rest away.

    Args:
        coverage: The tile's instrument sets, in any order.
        grid: The grid the tile is searched over.
        criteria: The filter read against the tile, holding the pixel floors.

    Returns:
        held: What the tile keeps, with the set each belongs to and the cells it
            fills.
        refused: What it turned away, carrying the same.
    """
    held: Held = []
    refused: Held = []
    least = criteria.least
    for owner, instrument in enumerate(coverage):
        for observation in instrument.events:
            spread, pixels = observation.own_km2, observation.pixels
            cells = [
                cell
                for cell in packing.cells_of(observation.mask).tolist()
                if cell in grid.inside
            ]
            if not cells:
                continue
            landed = (
                pixels * len(cells) * grid.cell_km2 / spread
                if spread and pixels is not None
                else 0.0
            )
            taken = held if landed >= least[owner] else refused
            taken.append((observation, owner, cells))
    return held, refused
