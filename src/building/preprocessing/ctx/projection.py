"""Placing one CTX scan on the grid its label projects it onto."""

from __future__ import annotations

import numpy as np

# The only projection ASU writes a CTX RDR in.
PROJECTION = "SimpleCylindrical"

# The only reading of latitude, longitude and their range this places.
CONVENTIONS = {
    "LatitudeType": "Planetocentric",
    "LongitudeDirection": "PositiveEast",
    "LongitudeDomain": "360",
}


def load(label: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    """Return the centre latitude of every line and longitude of every sample.

    Args:
        label: The parsed ISIS label of one scan.

    Returns:
        The latitude of every line, falling southward, and the longitude of
        every sample, rising eastward, both in degrees.

    Raises:
        ValueError: When the label names a projection or a convention this
            cannot read.
    """
    if label["ProjectionName"] != PROJECTION:
        raise ValueError(f"Cannot place a {label['ProjectionName']} grid.")
    for key, wanted in CONVENTIONS.items():
        if label[key] != wanted:
            raise ValueError(f"Cannot place a grid whose {key} is {label[key]}.")
    # The corner the projection starts from, in metres east and north of it.
    radius = float(label["EquatorialRadius"])
    half = float(label["PixelResolution"]) / 2.0
    # How many degrees one pixel spans, the same in both directions.
    step = float(np.degrees(float(label["PixelResolution"]) / radius))
    north = np.degrees((float(label["UpperLeftCornerY"]) - half) / radius)
    west = float(label["CenterLongitude"]) + np.degrees(
        (float(label["UpperLeftCornerX"]) + half) / radius
    )
    return (
        north - np.arange(int(label["Lines"])) * step,
        west + np.arange(int(label["Samples"])) * step,
    )
