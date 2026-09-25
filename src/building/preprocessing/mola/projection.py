"""The grid one MOLA product is projected onto, as its label writes it."""

from __future__ import annotations

import numpy as np

from building.preprocessing.common.models.samples import Samples
from common.maths import physics

# The two projections the gridded record is written in.
EQUATORIAL = "SIMPLE CYLINDRICAL"
POLAR = "POLAR STEREOGRAPHIC"


def grid_samples(label: dict[str, str]) -> Samples:
    """Return where every line and every sample of one product sits.

    Args:
        label: The parsed label of one product.

    Returns:
        samples: The latitude or northing of every line, the longitude or easting of
            every sample, and the polar grid they are measured on, if any.

    Raises:
        ValueError: When the label names a projection this cannot read.
    """
    named = label["MAP_PROJECTION_TYPE"]
    # How fine the grid is, which both projections count in bins to the degree.
    resolution = float(label["MAP_RESOLUTION"])
    lines, samples = int(label["LINES"]), int(label["LINE_SAMPLES"])
    if named == POLAR:
        # A cap is placed from its middle, in the stereographic metres MAP_SCALE names
        radius = float(label["A_AXIS_RADIUS"]) * physics.METRES_PER_KM
        down = np.radians((lines / 2.0 - 0.5 - np.arange(lines)) / resolution) * radius
        across = (
            np.radians((np.arange(samples) - samples / 2.0 + 0.5) / resolution) * radius
        )
        polar = (0.0, float(label["CENTER_LATITUDE"]) > 0.0, radius)
        return Samples(down, across, True, polar)
    if named != EQUATORIAL:
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
    return Samples(
        north - np.arange(lines) * step, west + np.arange(samples) * step, True, None
    )
