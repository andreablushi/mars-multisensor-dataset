"""Reading the written filter against one feature, into what it asks of that one."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence

from analysis.coverage import ground
from analysis.coverage.models.coverage import SetCoverage
from analysis.selector.models.filter import Constraints, Filter
from analysis.selector.models.grid import Grid


def clean_window(
    criteria: Filter, coverage: Sequence[SetCoverage], grid: Grid
) -> Filter:
    """Settle everything the written filter asks of one feature.

    Args:
        criteria: What the instruments are asked for, and which of them are timeless.
        coverage: The feature's instrument sets, in any order.
        grid: The grid the feature is searched over.

    Returns:
        criteria: The same filter, carrying what it asks of the feature.
    """
    iids = [instrument.summary.iid for instrument in coverage]
    windowed: Constraints = []
    standing: Constraints = []
    for constraint in criteria.constraints:
        answers = [
            (
                tuple(index for index, owner in enumerate(iids) if owner == iid),
                max(1, math.ceil(ground.cells(share, grid.area_km2, grid.cell_km2))),
            )
            for iid, share in constraint.items()
        ]
        # A constraint is out of the window only when everything answering it is
        held = (
            standing
            if all(iid in criteria.timeless for iid in constraint)
            else windowed
        )
        held.append(answers)
    # The whole-grid bar is scaled by the ground held, by one axis or by both
    covered = grid.area_km2 / (grid.cells * grid.cell_km2)
    return dataclasses.replace(
        criteria,
        least=[
            criteria.admits.get(iid, 0.0)
            * (
                # A set publishing a swath width is a sounder, its pixels on a line
                math.sqrt(covered)
                if any(one.width_km is not None for one in instrument.events)
                else covered
            )
            for iid, instrument in zip(iids, coverage, strict=True)
        ],
        windowed=sorted(windowed, key=lambda answers: -min(f for _, f in answers)),
        standing=sorted(standing, key=lambda answers: -min(f for _, f in answers)),
    )
