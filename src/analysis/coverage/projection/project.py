"""The projection of one instrument set's stored footprints onto every tile."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from shapely import STRtree, from_wkt, is_missing
from shapely.geometry.base import BaseGeometry

from analysis.coverage.models.observation import (
    ProjectedObservation,
    ProjectedSet,
)
from analysis.coverage.projection import footprints, size
from analysis.models.observation import ObservationSet
from common.maths import physics
from common.models.tile import Tile


def project_every_tile(
    stored: ObservationSet, tiles: Sequence[Tile]
) -> tuple[list[ProjectedSet], int]:
    """Project one set's footprints onto every tile they reach and cut them to it.

    Args:
        stored: The set's stored observations over one tile group, in time order.
        tiles: The tiles of that group.

    Returns:
        projected: One set per tile an observation landed on, in tile order.
        discarded: How many stored records could not be measured or reached no tile.
    """
    observations = stored.observations
    if not observations:
        return [], stored.discarded
    geoms = from_wkt(
        np.asarray([observation.wkt for observation in observations], dtype=object)
    )
    widths_m = size.track_widths(observations, geoms)
    swath_widths_m = np.asarray([width or 0.0 for width in widths_m], dtype=float)
    index = STRtree(geoms)
    polar = {
        north: from_wkt(
            np.asarray([getattr(observation, key) for observation in observations])
        )
        for north, key in ((True, "north_wkt"), (False, "south_wkt"))
    }
    polar_index = {north: STRtree(shapes) for north, shapes in polar.items()}
    reached = np.zeros(len(observations), dtype=bool)
    projected: list[ProjectedSet] = []
    for tile in tiles:
        region = footprints.tile_region(tile)
        near = np.sort(index.query(region.wide))
        batches = []
        if region.polar is not None:
            polar_geoms = polar[region.north]
            polar_near = np.sort(polar_index[region.north].query(region.polar_wide))
            near = near[is_missing(polar_geoms[near])]
            batches.append((polar_near, polar_geoms, True))
        batches.append((near, geoms, False))
        reaching: list[tuple[int, BaseGeometry]] = []
        for positions, shapes, stereographic in batches:
            if positions.size:
                reaching += zip(
                    positions.tolist(),
                    footprints.projected_footprints(
                        region,
                        shapes[positions],
                        swath_widths_m[positions],
                        stereographic=stereographic,
                    ),
                    strict=True,
                )
        landed = []
        for position, shape in sorted(reaching, key=lambda pair: pair[0]):
            if shape.is_empty:
                continue
            reached[position] = True
            observation, width_m = observations[position], widths_m[position]
            width_km = width_m / physics.METRES_PER_KM if width_m is not None else None
            landed.append(
                ProjectedObservation(
                    pdsid=observation.pdsid,
                    ihid=observation.ihid,
                    iid=observation.iid,
                    pt=observation.pt,
                    start=observation.start,
                    shape=shape,
                    width_km=width_km,
                    pixel_km2=size.ground_pixel_km2(
                        stored.set_key, observation.map_scale_m, width_km
                    ),
                )
            )
        if landed:
            projected.append(
                ProjectedSet(
                    tile=tile,
                    set_key=stored.set_key,
                    region=region,
                    observations=landed,
                )
            )
    return projected, stored.discarded + int(np.count_nonzero(~reached))
