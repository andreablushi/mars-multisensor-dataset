"""The footprints as ODE published them, read back off the downloaded metadata."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from shapely import wkt as reading
from shapely.geometry.base import BaseGeometry

from analysis import paths
from analysis.metadata.loaders.observations import load_observations
from analysis.models.instrument import InstrumentSet
from analysis.visualization.common.models.coverage import Coverage
from analysis.visualization.feature.models.outlines import Trace
from shared.disk.slugify import slugify

OUTLINE_CACHE = 4


def read(coverage: Coverage) -> dict[str, BaseGeometry]:
    """Read the published footprint of every observation of one feature."""
    summary = coverage[0].summary
    found: dict[str, BaseGeometry] = {}
    for instrument in coverage:
        found.update(
            _published(
                summary.feature_class, summary.feature_name, instrument.summary.set_key
            )
        )
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
def _published(feature_class: str, name: str, set_key: str) -> dict[str, BaseGeometry]:
    """Read one instrument set's published footprints, held for its panels."""
    path = (
        paths.METADATA_ROOT
        / slugify(feature_class)
        / slugify(name)
        / f"{InstrumentSet.from_key(set_key).slug}.jsonl"
    )
    return {
        observation.pdsid: reading.loads(observation.wkt)
        for observation in load_observations(path).observations
    }
