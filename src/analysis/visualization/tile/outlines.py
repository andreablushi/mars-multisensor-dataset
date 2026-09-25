"""The footprints as ODE published them, read back off the downloaded metadata."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from shapely import wkt
from shapely.geometry.base import BaseGeometry

from analysis import paths
from analysis.metadata.loaders.observations import load_observations
from analysis.models.instrument import InstrumentSet
from analysis.utils.tile_group import group_of, tile_grid
from analysis.visualization.panels import Coverage


def read_footprints(coverage: Coverage) -> dict[str, BaseGeometry]:
    """Read the published footprint of every observation of one tile.

    Args:
        coverage: The tile's instrument sets.

    Returns:
        footprints: Each observed set's published footprints, by pdsid.
    """
    group = group_of(tile_grid().tile_named(coverage[0].summary.tile))
    footprints: dict[str, BaseGeometry] = {}
    for instrument in coverage:
        if instrument.observed:
            footprints.update(_set_footprints(group, instrument.summary.set_key))
    return footprints


def footprint_lines(footprint: BaseGeometry) -> list[tuple[np.ndarray, np.ndarray]]:
    """Trace one published footprint as the lines a panel can draw.

    Args:
        footprint: The footprint, a single shape or a collection of them.

    Returns:
        lines: The longitudes and latitudes of each part's outer ring or line.
    """
    lines = []
    for part in getattr(footprint, "geoms", [footprint]):
        coordinates = np.asarray(getattr(part, "exterior", part).coords, dtype=float)
        lines.append((coordinates[:, 0], coordinates[:, 1]))
    return lines


@lru_cache(maxsize=4)
def _set_footprints(group: str, set_key: str) -> dict[str, BaseGeometry]:
    """Read one instrument set's published footprints, held for its panels.

    Args:
        group: The tile group the set was downloaded for.
        set_key: The set's key.

    Returns:
        footprints: Its published footprints, by pdsid.
    """
    path = paths.metadata_path(group, InstrumentSet.from_key(set_key))
    return {
        observation.pdsid: wkt.loads(observation.wkt)
        for observation in load_observations(path).observations
    }
