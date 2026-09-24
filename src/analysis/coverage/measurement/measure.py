"""Measuring how one instrument set covers one tile through time."""

from __future__ import annotations

import numpy as np

from analysis.coverage.measurement.accumulation import fine_split, union
from analysis.coverage.models.coverage import Event
from analysis.coverage.models.observation import ProjectedSet
from analysis.coverage.models.summary import Summary
from analysis.utils import mask as packing


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
    ground = union.new_ground(
        region, [observation.shape for observation in observations], union_threads
    )
    cumulative = np.cumsum(ground)
    grid = fine_split.grid_over(region, grid_cells)
    events = [
        Event(
            tile=tile.name,
            ihid=observation.ihid,
            iid=observation.iid,
            pt=observation.pt,
            pdsid=observation.pdsid,
            t_start=observation.start,
            own_km2=observation.shape.area / 1e6,
            cum_frac=float(covered) / region.area_m2,
            width_km=observation.width_km,
            pixels=observation.shape.area / 1e6 / observation.pixel_km2,
            mask=packing.encode(
                fine_split.filled_cells(grid, observation.shape), grid.side**2
            ),
        )
        for observation, covered in zip(observations, cumulative, strict=True)
    ]
    return events, Summary(
        tile=tile.name,
        set_key=projected.set_key,
        ihid=events[0].ihid,
        iid=events[0].iid,
        pt=events[0].pt,
        tile_area_km2=region.area_m2 / 1e6,
        covered_frac=float(cumulative[-1]) / region.area_m2,
        n_obs=len(events),
        t_first=events[0].t_start,
        t_last=events[-1].t_start,
        pixels=sum(event.pixels for event in events),
        grid_side=grid.side,
        cell_km2=grid.cell_area_m2 / 1e6,
        grid_mask=packing.encode(
            fine_split.filled_cells(grid, region.shape), grid.side**2
        ),
    )
