"""The projection of one instrument set's stored footprints onto every tile."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from shapely import STRtree, from_wkt, is_missing
from shapely.geometry.base import BaseGeometry

from analysis.coverage.models.footprints import ParsedFootprints
from analysis.coverage.models.observation import (
    ProjectedObservation,
    ProjectedSet,
)
from analysis.coverage.models.region import TileRegion
from analysis.coverage.projection import footprints
from analysis.models.instrument import SHARAD_SWATH_M
from analysis.models.observation import Observation, ObservationSet
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
    parsed = parsed_footprints(observations)
    reached = np.zeros(len(observations), dtype=bool)
    projected: list[ProjectedSet] = []
    for tile in tiles:
        region = footprints.tile_region(tile)
        landed: dict[int, BaseGeometry] = {}
        for positions, shapes, stereographic in footprints_near(parsed, region):
            laea = footprints.laea_footprints(
                region,
                shapes[positions],
                parsed.swath_widths_m[positions],
                stereographic=stereographic,
            )
            landed.update(
                (position, shape)
                for position, shape in zip(positions.tolist(), laea, strict=True)
                if not shape.is_empty
            )
        if landed:
            reached[list(landed)] = True
            projected.append(
                ProjectedSet(
                    tile=tile,
                    set_key=stored.set_key,
                    region=region,
                    observations=[
                        projected_observation(observations[position], shape)
                        for position, shape in sorted(landed.items())
                    ],
                )
            )
    return projected, stored.discarded + int(np.count_nonzero(~reached))


def parsed_footprints(observations: Sequence[Observation]) -> ParsedFootprints:
    """Parse every footprint of one set, in lon/lat and on each pole, and index them.

    Args:
        observations: The set's observations.

    Returns:
        parsed: The footprints, their search trees, and each track's swath width.
    """
    return ParsedFootprints(
        geographic=STRtree(from_wkt([observation.wkt for observation in observations])),
        polar={
            north: STRtree(
                from_wkt([getattr(observation, key) for observation in observations])
            )
            for north, key in ((True, "north_wkt"), (False, "south_wkt"))
        },
        swath_widths_m=np.asarray(
            [
                SHARAD_SWATH_M if observation.is_track else 0.0
                for observation in observations
            ]
        ),
    )


def footprints_near(
    parsed: ParsedFootprints, region: TileRegion
) -> list[tuple[np.ndarray, np.ndarray, bool]]:
    """Find the footprints that may reach one tile, polar ones wherever ODE has them.

    Args:
        parsed: The set's footprints and their search trees.
        region: The projected tile and its clipping regions.

    Returns:
        batches: Each batch's positions, the footprints they index, and whether those
            are in polar stereographic metres.
    """
    near = parsed.geographic.query(region.clip_wide)
    batches = []
    if region.polar_clip is not None:
        polar = parsed.polar[region.north]
        near = near[is_missing(polar.geometries[near])]
        batches.append((polar.query(region.polar_clip_wide), polar.geometries, True))
    batches.append((near, parsed.geographic.geometries, False))
    return [batch for batch in batches if batch[0].size]


def projected_observation(
    observation: Observation, shape: BaseGeometry
) -> ProjectedObservation:
    """Return one observation carrying the ground it covers on the tile."""
    return ProjectedObservation(
        pdsid=observation.pdsid,
        ihid=observation.ihid,
        iid=observation.iid,
        pt=observation.pt,
        start=observation.start,
        shape=shape,
    )
