"""Every instrument's observations of one tile, merged onto one time axis."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from analysis.coverage.models.coverage import SetCoverage
from analysis.selector.filters.admit import admitted_observations
from analysis.selector.filters.coverage_constraints import tile_constraints
from analysis.selector.models.criteria import Criteria
from analysis.selector.models.search_grid import SearchGrid
from analysis.selector.models.track import Track
from analysis.selector.solar_longitude import solar_longitude
from analysis.utils import mask as packing

# Seconds in a day, which is what every span is measured in.
DAY_SECONDS = 86400.0


def merge_track(coverage: Sequence[SetCoverage], criteria: Criteria) -> Track | None:
    """Merge a tile's instrument sets onto one timeline, under what the criteria ask.

    Args:
        coverage: The tile's instrument sets, in any order.
        criteria: Which instruments a window has to hold, and how much ground each.

    Returns:
        track: The timeline, or None when the tile holds nothing measurable.
    """
    summary = coverage[0].summary
    inside = packing.filled_cells(summary.grid_mask).tolist()
    grid = SearchGrid(
        cell_count=summary.grid_side * summary.grid_side,
        area_km2=len(inside) * summary.cell_km2,
        cell_km2=summary.cell_km2,
        inside=frozenset(inside),
    )
    # The one place the constraints are read, which everything below takes them from
    windowed, standing = tile_constraints(criteria, coverage, grid)
    admitted, refused = admitted_observations(coverage, grid, criteria)
    if not admitted:
        return None
    admitted.sort(key=lambda offered: offered[0].t_start)
    times = [
        observation.t_start.timestamp() / DAY_SECONDS for observation, _, _ in admitted
    ]
    return Track(
        observations=[observation for observation, _, _ in admitted],
        times=times,
        ls=[solar_longitude(day) for day in times],
        owners=[owner for _, owner, _ in admitted],
        cells=[np.asarray(cells, dtype=np.intp) for _, _, cells in admitted],
        labels=[instrument.label for instrument in coverage],
        iids=[instrument.summary.iid for instrument in coverage],
        grid=grid,
        refused=sorted(refused, key=lambda offered: offered[0].t_start),
        windowed=windowed,
        standing=standing,
    )
