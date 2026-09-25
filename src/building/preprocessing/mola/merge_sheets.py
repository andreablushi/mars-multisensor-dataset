"""Laying the sheets one tile stands on onto the single grid they share."""

from __future__ import annotations

import math

import numpy as np

from building.preprocessing.common.models.position import Position
from building.preprocessing.mola import projection
from building.preprocessing.mola.models.observation import MolaObservation
from common.maths import geodesy
from common.maths.geodesy import TURN
from common.models.tile import Tile
from common.pds import images, labels


def merge_sheets(
    observation: MolaObservation, frame: Tile
) -> tuple[dict[str, str], np.ndarray, Position]:
    """Return the one grid every sheet a tile stands on writes its part of.

    Args:
        observation: The sheets of the grid that landed, and how fine it is.
        frame: The local frame of the tile the sheets are merged for.

    Returns:
        label: What the sheets that write part of the box say about it, merged.
        height: The height above the areoid in metres over the box, lines by samples.
        position: The latitude of every line, falling southward, and the longitude of
            every sample, rising eastward past a turn.

    Raises:
        FileNotFoundError: When a sheet's label is missing.
        KeyError: When a label names a sample type this cannot read.
        ValueError: When the projection is unreadable or the box is not covered.
    """
    resolution = observation.grid.resolution
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
    for _, image in sorted(observation.files.items()):
        label = labels.load(image.with_suffix(".lbl"))
        placed = projection.grid_position(label)
        # Where the sheet's own first bin sits on the grid every sheet shares.
        line = round((90.0 - float(placed.north[0])) * resolution - 0.5)
        sample = round(float(placed.east[0]) * resolution - 0.5) % whole
        lines, samples = placed.sizes
        top, bottom = max(down.start, line), min(down.stop, line + lines)
        touched = False
        # A box running over the meridian meets a sheet a whole turn along, too.
        for west in (sample, sample + whole):
            left, right = max(across.start, west), min(across.stop, west + samples)
            if top >= bottom or left >= right:
                continue
            part = images.load_window(
                image, label, (top - line, bottom - line), (left - west, right - west)
            )
            if height is None:
                height = np.zeros((len(down), len(across)), dtype=part.dtype)
            at = np.s_[
                top - down.start : bottom - down.start,
                left - across.start : right - across.start,
            ]
            height[at] = part
            written[at] = True
            touched = True
        if touched:
            read.append(label)
    if height is None or not written.all():
        raise ValueError(
            f"{frame.name} reaches ground no sheet of {observation.identifier} holds."
        )
    return (
        labels.merge(*read),
        height,
        Position(
            90.0 - (np.arange(down.start, down.stop) + 0.5) / resolution,
            (np.arange(across.start, across.stop) + 0.5) / resolution,
            True,
        ),
    )
