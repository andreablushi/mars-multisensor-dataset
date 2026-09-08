"""Laying the tiles one feature stands on onto the single grid they share."""

from __future__ import annotations

import math

import numpy as np

from building.common.pds import images, labels
from building.models.feature import FeatureFrame
from building.preprocessing.mola import projection
from building.preprocessing.mola.models.grid import MolaGrid
from building.preprocessing.mola.models.observation import MolaObservation
from shared.geometry import geodesy
from shared.geometry.geodesy import TURN


def merge_tiles(grid: MolaGrid, frame: FeatureFrame) -> MolaObservation:
    """Return the one grid every tile a feature stands on writes its part of.

    Args:
        grid: The tiles of the grid that landed, and how fine it is.
        frame: The local frame of the feature the tiles are merged for.

    Returns:
        observation: The observation holding that feature's own box and no more of the
            grid.

    Raises:
        FileNotFoundError: When a tile's label is missing.
        KeyError: When a label names a sample type this cannot read.
        ValueError: When a label names a projection this cannot read, or the
            tiles that landed leave any part of the box unwritten.
    """
    resolution = grid.resolution
    whole = round(TURN) * resolution
    # Which bins the box covers: lines south from the pole, samples east of it
    span = geodesy.longitude_span(frame.west_lon, frame.east_lon)
    down = range(
        math.ceil((90.0 - frame.max_lat) * resolution - 0.5),
        math.floor((90.0 - frame.min_lat) * resolution - 0.5) + 1,
    )
    across = range(
        math.ceil(frame.west_lon * resolution - 0.5),
        math.floor((frame.west_lon + span) * resolution - 0.5) + 1,
    )
    height: np.ndarray | None = None
    written = np.zeros((len(down), len(across)), dtype=bool)
    read = []
    for tile, image in sorted(grid.files.items()):
        label = labels.load(image.with_suffix(".lbl"))
        read.append(label)
        latitude, longitude, _ = projection.grid_axes(label)
        # Where the tile's own first bin sits on the grid every tile shares.
        line = round((90.0 - float(latitude[0])) * resolution - 0.5)
        sample = round(float(longitude[0]) * resolution - 0.5) % whole
        lines, samples = int(label["LINES"]), int(label["LINE_SAMPLES"])
        # A box running over the meridian meets a tile a whole turn along, too.
        for shift in (0, whole):
            first, last = max(down.start, line), min(down.stop, line + lines)
            starts = max(across.start, sample + shift)
            stops = min(across.stop, sample + shift + samples)
            if first >= last or starts >= stops:
                continue
            part = images.load_window(
                image,
                label,
                (first - line, last - line),
                (starts - sample - shift, stops - sample - shift),
            )
            if height is None:
                height = np.zeros((len(down), len(across)), dtype=part.dtype)
            at = np.s_[
                first - down.start : last - down.start,
                starts - across.start : stops - across.start,
            ]
            height[at] = part
            written[at] = True
    if height is None or not written.all():
        raise ValueError(
            f"{frame.feature_name} reaches ground no tile of {grid.name} holds."
        )
    return MolaObservation(
        grid.name,
        labels.merge(*read),
        height,
        90.0 - (np.arange(down.start, down.stop) + 0.5) / resolution,
        (np.arange(across.start, across.stop) + 0.5) / resolution,
    )
