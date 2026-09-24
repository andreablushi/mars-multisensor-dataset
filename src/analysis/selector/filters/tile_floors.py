"""The floors the written filter sets one tile: its pixels per set, its constraints."""

from __future__ import annotations

import math
from collections.abc import Sequence

from analysis.coverage.models.coverage import SetCoverage
from analysis.selector.models.filter import Constraints, Filter
from analysis.selector.models.search_grid import SearchGrid


def tile_floors(
    criteria: Filter, coverage: Sequence[SetCoverage], grid: SearchGrid
) -> tuple[list[float], Constraints, Constraints]:
    """Settle everything the written filter asks of one tile.

    Args:
        criteria: What the instruments are asked for, and which of them are timeless.
        coverage: The tile's instrument sets, in any order.
        grid: The grid the tile is searched over.

    Returns:
        min_pixels: The pixels each set has to land on the tile, by set.
        windowed: What a window is scored on, tightest constraint first.
        standing: What the whole record answers for, tightest first.
    """
    iids = [instrument.summary.iid for instrument in coverage]
    windowed: Constraints = []
    standing: Constraints = []
    for constraint in criteria.constraints:
        answers = [
            (
                tuple(index for index, owner in enumerate(iids) if owner == iid),
                max(1, math.ceil(share * grid.area_km2 / grid.cell_km2)),
            )
            for iid, share in constraint.items()
        ]
        # A constraint is out of the window only when everything answering it is
        timeless = all(iid in criteria.timeless for iid in constraint)
        (standing if timeless else windowed).append(answers)
    for constraints in (windowed, standing):
        constraints.sort(key=lambda answers: -min(floor for _, floor in answers))
    # The whole-grid bar is scaled by the ground held, by one axis or by both
    inside_share = grid.area_km2 / (grid.cell_count * grid.cell_km2)
    min_pixels = [
        criteria.admits.get(iid, 0.0)
        * (
            # A set publishing a swath width is a sounder, its pixels on a line
            math.sqrt(inside_share)
            if any(event.width_km is not None for event in instrument.events)
            else inside_share
        )
        for iid, instrument in zip(iids, coverage, strict=True)
    ]
    return min_pixels, windowed, standing
