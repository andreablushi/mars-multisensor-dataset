"""Placing one MOLA tile on the grid its label projects it onto."""

from __future__ import annotations

import numpy as np

# The only projection the gridded record is written in.
PROJECTION = "SIMPLE CYLINDRICAL"


def load(label: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    """Return the centre latitude of every line and longitude of every sample.

    A simple cylindrical grid is even in both directions, so the two axes are
    all that place it and a pixel sits where they cross.

    Args:
        label: The parsed label of one plane.

    Returns:
        The latitude of every line, falling southward, and the longitude of
        every sample, rising eastward, both in degrees.

    Raises:
        ValueError: When the label names a projection this cannot read.
    """
    if label["MAP_PROJECTION_TYPE"] != PROJECTION:
        raise ValueError(f"Cannot place a {label['MAP_PROJECTION_TYPE']} grid.")
    # How many degrees one pixel spans, the same in both directions.
    step = 1.0 / float(label["MAP_RESOLUTION"])
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
        north - np.arange(int(label["LINES"])) * step,
        west + np.arange(int(label["LINE_SAMPLES"])) * step,
    )
