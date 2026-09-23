"""What one track's geometry says of the Sun and the ionosphere over each group."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from analysis.models.tile_group import TileGroup

SOLAR_ZENITH_FIELD = "Solar_zenith_angle"
PHASE_FIELD = "Phase_distortion"

# The geometry table's columns, counted from nought: latitude, longitude, SZA, phase
COLUMNS = (2, 3, 8, 9)

# One group's median solar zenith angle and phase distortion, or None
Measured = list[float] | None


def measure_track(path: Path, groups: Mapping[str, TileGroup]) -> dict[str, Measured]:
    """Read the median SZA and phase of the traces one track sounded over each group.

    Args:
        path: The track's geometry table, one row per radargram column.
        groups: The groups whose records list the track, by name.

    Returns:
        measured: The medians over each group's box, None where no trace fell in it.
    """
    latitude, longitude, zenith, phase = np.loadtxt(
        path, delimiter=",", usecols=COLUMNS, unpack=True, ndmin=2
    )
    measured: dict[str, Measured] = {}
    for name, group in groups.items():
        inside = (latitude >= group.min_lat) & (latitude <= group.max_lat)
        if not group.circles_a_pole:
            east = longitude % 360.0
            inside &= (east >= group.west_lon) & (east <= group.east_lon)
        measured[name] = (
            [
                round(float(np.median(zenith[inside])), 2),
                round(float(np.median(phase[inside])), 3),
            ]
            if inside.any()
            else None
        )
    return measured
