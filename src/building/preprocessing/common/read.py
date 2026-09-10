"""Reading one crop back out of the store, placed where its feature stands."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from building.preprocessing.common import relative_positioning
from building.preprocessing.common.models.relative_position import RelativePosition
from building.preprocessing.common.store import EAST, META, NORTH
from shared.models.feature import Feature


def read_sample(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    """Read one stored crop back, its arrays and what describes them.

    Args:
        path: The file `write_sample` wrote it as.

    Returns:
        arrays: Every array it holds, by the name it was written as.
        described: What the crop is, its axes, its feature and its units.
    """
    with np.load(path) as held:
        described = json.loads(str(held[META]))
        arrays = {name: held[name] for name in held.files if name != META}
    return arrays, described


def ground_metres(
    arrays: dict[str, np.ndarray], described: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Return how far north and east of its feature centre every sample sits.

    Args:
        arrays: The crop's arrays, holding the two the position was written as.
        described: What `read_sample` handed back beside them.

    Returns:
        north: The ground metres north of that centre, one per sample, in the
            azimuthal equidistant frame it is the middle of.
        east: The ground metres east of it, in the same frame.
    """
    grid = described["polar"]
    position = RelativePosition(
        north=arrays[NORTH],
        east=arrays[EAST],
        separable=described["separable"],
        polar=None if grid is None else tuple(grid),
    )
    frame = Feature(
        described["feature_class"], described["feature_name"], **described["box"]
    )
    return relative_positioning.ground_metres(position, frame)
