"""The coverage a tile is asked for, and how much ground a window answers it with."""

from __future__ import annotations

import math
from collections.abc import Sequence

from analysis.coverage.models.coverage import SetCoverage
from analysis.selector.models.criteria import Constraints, Criteria
from analysis.selector.models.search_grid import SearchGrid


def cells_per_constraint(
    constraints: Constraints, cells_reached: Sequence[int]
) -> list[int] | None:
    """Take what each constraint reaches, or refuse them all when one goes unmet.

    Args:
        constraints: The sets answering each one and their floors, tightest first.
        cells_reached: How many cells each set reaches inside the window.

    Returns:
        counts: The cells each constraint reaches, in order, or None when one is unmet.
    """
    counts: list[int] = []
    for answers in constraints:
        # A constraint is answered by whichever instrument reaches most of its bar
        cell_count = 0
        for answering, floor in answers:
            reached = max((cells_reached[owner] for owner in answering), default=0)
            if reached >= floor and reached > cell_count:
                cell_count = reached
        if not cell_count:
            return None
        counts.append(cell_count)
    return counts


def tile_constraints(
    criteria: Criteria, coverage: Sequence[SetCoverage], grid: SearchGrid
) -> tuple[Constraints, Constraints]:
    """Turn every coverage constraint into the cells it asks of one tile.

    Args:
        criteria: What the instruments are asked for, and which of them are timeless.
        coverage: The tile's instrument sets, in any order.
        grid: The grid the tile is searched over.

    Returns:
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
    return windowed, standing
