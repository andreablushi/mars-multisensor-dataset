"""What one instrument set covers of one tile, as its event rows and summary row."""

from __future__ import annotations

import numpy as np

from analysis.coverage.measurement import cells
from analysis.coverage.models.coverage import Event
from analysis.coverage.models.observation import ProjectedSet
from analysis.coverage.models.summary import Summary
from analysis.utils import mask as packing
from common.maths import physics

M2_PER_KM2 = physics.METRES_PER_KM**2


def measure_set(projected: ProjectedSet, cell_km: float) -> tuple[list[Event], Summary]:
    """Measure how one instrument set covers one tile over time.

    Args:
        projected: The set's ground on the tile, in chronological order.
        cell_km: The side of one coverage cell, in kilometres.

    Returns:
        events: One row per observation.
        summary: The single row describing the set.
    """
    tile, region = projected.tile, projected.region
    observations = projected.observations
    grid = cells.cell_grid(region, cell_km)
    tile_cells = cells.filled_cells(grid, region.laea)
    union_cells = np.zeros(grid.side**2, dtype=bool)
    events = []
    for observation in observations:
        filled = cells.filled_cells(grid, observation.shape)
        union_cells[filled] = True
        own_km2 = observation.shape.area / M2_PER_KM2
        events.append(
            Event(
                tile=tile.name,
                ihid=observation.ihid,
                iid=observation.iid,
                pt=observation.pt,
                pdsid=observation.pdsid,
                t_start=observation.start,
                own_km2=own_km2,
                cum_frac=np.count_nonzero(union_cells[tile_cells]) / tile_cells.size,
                mask=packing.packed_mask(filled, grid.side**2),
            )
        )
    return events, Summary(
        tile=tile.name,
        set_key=projected.set_key,
        iid=events[0].iid,
        tile_area_km2=region.laea.area / M2_PER_KM2,
        covered_frac=events[-1].cum_frac,
        n_obs=len(events),
        t_first=events[0].t_start,
        t_last=events[-1].t_start,
        grid_side=grid.side,
        cell_km2=grid.cell_area_m2 / M2_PER_KM2,
        grid_mask=packing.packed_mask(tile_cells, grid.side**2),
    )
