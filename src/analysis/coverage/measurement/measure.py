"""What one instrument set covers of one tile, as its event rows and summary row."""

from __future__ import annotations

import numpy as np

from analysis.coverage.measurement.accumulation import fine_split, union
from analysis.coverage.models.coverage import Event
from analysis.coverage.models.observation import ProjectedSet
from analysis.coverage.models.summary import Summary
from analysis.utils import mask as packing
from common.maths import physics

M2_PER_KM2 = physics.METRES_PER_KM**2


def measure_set(
    projected: ProjectedSet, grid_cells: int, union_threads: int
) -> tuple[list[Event], Summary]:
    """Measure how one instrument set covers one tile over time.

    Args:
        projected: The set's ground on the tile, in chronological order.
        grid_cells: How many cells one stretch of the tile's grid holds per axis.
        union_threads: How many of the tile's cells to accumulate at once.

    Returns:
        events: One row per observation.
        summary: The single row describing the set.
    """
    tile, region = projected.tile, projected.region
    observations = projected.observations
    tile_m2 = region.shape.area
    ground = union.new_ground(
        region, [observation.shape for observation in observations], union_threads
    )
    cumulative = np.cumsum(ground)
    grid = fine_split.grid_over(region, grid_cells)
    events = []
    for observation, covered in zip(observations, cumulative, strict=True):
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
                cum_frac=float(covered) / tile_m2,
                width_km=observation.width_km,
                pixels=own_km2 / observation.pixel_km2,
                mask=packing.packed_mask(
                    fine_split.filled_cells(grid, observation.shape), grid.side**2
                ),
            )
        )
    return events, Summary(
        tile=tile.name,
        set_key=projected.set_key,
        iid=events[0].iid,
        tile_area_km2=tile_m2 / M2_PER_KM2,
        covered_frac=float(cumulative[-1]) / tile_m2,
        n_obs=len(events),
        t_first=events[0].t_start,
        t_last=events[-1].t_start,
        grid_side=grid.side,
        cell_km2=grid.cell_area_m2 / M2_PER_KM2,
        grid_mask=packing.packed_mask(
            fine_split.filled_cells(grid, region.shape), grid.side**2
        ),
    )
