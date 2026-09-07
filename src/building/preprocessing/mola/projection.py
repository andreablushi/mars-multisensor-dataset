"""Placing one MOLA grid on the projection its own label writes it in."""

from __future__ import annotations

import numpy as np

from building.preprocessing.common.models.relative_position import PolarGrid

# The two projections the gridded record is written in.
CYLINDRICAL = "SIMPLE CYLINDRICAL"
POLAR = "POLAR STEREOGRAPHIC"

# The archive writes the sphere a cap is built on in kilometres.
KM = 1000.0


def load(label: dict[str, str]) -> tuple[np.ndarray, np.ndarray, PolarGrid | None]:
    """Return what places every line and every sample of one grid.

    Args:
        label: The parsed label of one product.

    Returns:
        What every line holds and what every sample holds, and the pole the two
        are measured on. A cylindrical grid gives the latitude of every line,
        falling southward, and the longitude of every sample, rising eastward,
        both in degrees, and no pole beside them. A cap gives the northing and
        the easting in the projection's own metres, and the pole that turns
        them back into degrees.

    Raises:
        ValueError: When the label names a projection this cannot read.
    """
    named = label["MAP_PROJECTION_TYPE"]
    # How fine the grid is, which both projections count in bins to the degree.
    resolution = float(label["MAP_RESOLUTION"])
    lines, samples = int(label["LINES"]), int(label["LINE_SAMPLES"])
    if named == POLAR:
        # A cap is placed from its own middle, and its degrees of arc are the
        # projection's metres on the sphere the archive built it on.
        radius = float(label["A_AXIS_RADIUS"]) * KM
        down = np.radians((lines / 2.0 - 0.5 - np.arange(lines)) / resolution) * radius
        across = (
            np.radians((np.arange(samples) - samples / 2.0 + 0.5) / resolution) * radius
        )
        return down, across, (0.0, float(label["CENTER_LATITUDE"]) > 0.0, radius)
    if named != CYLINDRICAL:
        raise ValueError(f"Cannot place a {named} grid.")
    # How many degrees one pixel spans, the same in both directions.
    step = 1.0 / resolution
    # The projection counts pixels from one, from the offset it puts its origin at.
    north = (
        float(label["CENTER_LATITUDE"])
        - (1.0 - float(label["LINE_PROJECTION_OFFSET"])) * step
    )
    west = (
        float(label["CENTER_LONGITUDE"])
        + (1.0 - float(label["SAMPLE_PROJECTION_OFFSET"])) * step
    )
    return (
        north - np.arange(lines) * step,
        west + np.arange(samples) * step,
        None,
    )
