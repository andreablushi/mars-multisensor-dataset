"""Every instrument's observations of one tile, merged onto one time axis."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from analysis.coverage.models.coverage import SetCoverage
from analysis.selector.filters import season
from analysis.selector.filters.admissible import admitted_observations
from analysis.selector.filters.tile_floors import tile_floors
from analysis.selector.models.filter import Filter
from analysis.selector.models.search_grid import SearchGrid
from analysis.selector.models.track import Track
from analysis.utils import mask as packing

# Seconds in a day, which is what every span is measured in.
DAY_SECONDS = 86400.0


def merge_track(coverage: Sequence[SetCoverage], criteria: Filter) -> Track | None:
    """Merge a tile's instrument sets onto one timeline, under what the filter asks.

    Args:
        coverage: The tile's instrument sets, in any order.
        criteria: Which instruments a window has to hold, and how much ground each.

    Returns:
        track: The timeline, or None when the tile holds nothing measurable.
    """
    summary = coverage[0].summary
    inside = packing.cells_of(summary.grid_mask).tolist()
    grid = SearchGrid(
        cells=summary.grid_side * summary.grid_side,
        area_km2=len(inside) * summary.cell_km2,
        cell_km2=summary.cell_km2,
        inside=frozenset(inside),
    )
    # The one place the filter is read, which everything below takes it from
    least, windowed, standing = tile_floors(criteria, coverage, grid)
    admitted, refused = admitted_observations(coverage, grid, least)
    if not admitted:
        return None
    admitted.sort(key=lambda offered: offered[0].t_start)
    times = [
        observation.t_start.timestamp() / DAY_SECONDS for observation, _, _ in admitted
    ]
    return Track(
        observations=[observation for observation, _, _ in admitted],
        times=times,
        ls=season.solar_longitudes(times),
        owners=[owner for _, owner, _ in admitted],
        cells=[np.asarray(cells, dtype=np.intp) for _, _, cells in admitted],
        labels=[instrument.label for instrument in coverage],
        iids=[instrument.summary.iid for instrument in coverage],
        grid=grid,
        refused=sorted(refused, key=lambda offered: offered[0].t_start),
        least=least,
        windowed=windowed,
        standing=standing,
    )
