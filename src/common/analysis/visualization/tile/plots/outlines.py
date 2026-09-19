"""The footprints as ODE published them, read back off the downloaded metadata."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from shapely import wkt as reading
from shapely.geometry.base import BaseGeometry

from common.analysis import configs, paths
from common.analysis.metadata.loaders.observations import load_observations
from common.analysis.models.instrument import InstrumentSet
from common.analysis.utils import tile_group
from common.analysis.visualization.common.models.coverage import Coverage
from common.analysis.visualization.tile.models.outlines import Trace
from common.maths.tessellate import Tessellate

OUTLINE_CACHE = 4


def read(coverage: Coverage) -> dict[str, BaseGeometry]:
    """Read the published footprint of every observation of one tile."""
    settings = configs.load()
    grid = Tessellate.of(settings.tile_km)
    group = tile_group.group_name(
        len(grid.columns),
        grid.tile_named(coverage[0].summary.tile),
        settings.tile_group_deg,
    )
    found: dict[str, BaseGeometry] = {}
    for instrument in coverage:
        if instrument.observed:
            found.update(_published(group, instrument.summary.set_key))
    return found


def traced(shape: BaseGeometry) -> list[Trace]:
    """Trace one published footprint as the lines a panel can draw."""
    drawn: list[Trace] = []
    for part in getattr(shape, "geoms", [shape]):
        ring = getattr(part, "exterior", None)
        line = ring if ring is not None else part
        coordinates = np.asarray(line.coords, dtype=float)
        drawn.append((coordinates[:, 0], coordinates[:, 1]))
    return drawn


@lru_cache(maxsize=OUTLINE_CACHE)
def _published(group: str, set_key: str) -> dict[str, BaseGeometry]:
    """Read one instrument set's published footprints, held for its panels."""
    path = paths.metadata_file(
        paths.METADATA_ROOT, group, InstrumentSet.from_key(set_key)
    )
    return {
        observation.pdsid: reading.loads(observation.wkt)
        for observation in load_observations(path).observations
    }
