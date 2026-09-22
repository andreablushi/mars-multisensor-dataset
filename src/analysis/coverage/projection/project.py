"""Putting one instrument set's stored footprints onto the ground of every tile."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from shapely import STRtree, from_wkt, is_missing
from shapely.geometry.base import BaseGeometry

from analysis.coverage.models.observation import (
    ProjectedObservation,
    ProjectedSet,
)
from analysis.coverage.projection.geometry import footprints, sizing
from analysis.models.observation import ObservationSet
from common.models.tile import Tile


def project_every_tile(
    loaded: ObservationSet, tiles: Sequence[Tile]
) -> tuple[list[ProjectedSet], int]:
    """Project one set's footprints onto every tile they reach and cut them to it.

    Args:
        loaded: The set's stored observations over one tile group, in time order.
        tiles: The tiles of that group.

    Returns:
        projected: One set per tile an observation landed on, in tile order.
        discarded: How many stored records could not be measured or reached no tile.
    """
    observations = loaded.observations
    if not observations:
        return [], loaded.discarded
    geoms = from_wkt(np.asarray([one.wkt for one in observations], dtype=object))
    widths = sizing.track_widths(observations)
    radii = np.asarray([width or 0.0 for width in widths], dtype=float)
    index = STRtree(geoms)
    polar = {
        north: from_wkt(np.asarray([getattr(one, key) for one in observations]))
        for north, key in ((True, "north_wkt"), (False, "south_wkt"))
    }
    polar_index = {north: STRtree(held) for north, held in polar.items()}
    reached = np.zeros(len(observations), dtype=bool)
    projected: list[ProjectedSet] = []
    for tile in tiles:
        region = footprints.tile_region(tile)
        near = np.sort(index.query(region.wide))
        reaching: list[tuple[int, BaseGeometry]] = []
        if region.polar is not None:
            held = polar[region.north]
            stereographic = np.sort(polar_index[region.north].query(region.polar_wide))
            near = near[is_missing(held[near])]
            if stereographic.size:
                reaching += zip(
                    stereographic.tolist(),
                    footprints.projected_footprints(
                        region, held[stereographic], radii[stereographic], True
                    ),
                    strict=True,
                )
        if near.size:
            reaching += zip(
                near.tolist(),
                footprints.projected_footprints(region, geoms[near], radii[near]),
                strict=True,
            )
        landed = []
        for at, shape in sorted(reaching, key=lambda pair: pair[0]):
            if shape.is_empty:
                continue
            reached[at] = True
            observation, width_m = observations[at], widths[at]
            width_km = width_m / 1000.0 if width_m is not None else None
            landed.append(
                ProjectedObservation(
                    pdsid=observation.pdsid,
                    ihid=observation.ihid,
                    iid=observation.iid,
                    pt=observation.pt,
                    start=observation.start,
                    stop=observation.stop,
                    shape=shape,
                    width_km=width_km,
                    pixel_km2=sizing.ground_pixel_km2(
                        loaded.set_key, observation.map_scale_m, width_km
                    ),
                )
            )
        if landed:
            projected.append(
                ProjectedSet(
                    tile=tile,
                    set_key=loaded.set_key,
                    region=region,
                    observations=landed,
                )
            )
    return projected, loaded.discarded + int(np.count_nonzero(~reached))
