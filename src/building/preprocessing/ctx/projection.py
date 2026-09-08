"""Placing one CTX scan on the grid its label projects it onto."""

from __future__ import annotations

import numpy as np

from building.preprocessing.common.models.relative_position import PolarGrid

# The two projections ASU writes a scan in, the second above about seventy degrees.
CYLINDRICAL = "SimpleCylindrical"
POLAR = "PolarStereographic"

# The only reading of latitude and longitude this places; the domain does not matter.
CONVENTIONS = {
    "LatitudeType": "Planetocentric",
    "LongitudeDirection": "PositiveEast",
}


def grid_axes(
    label: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, PolarGrid | None]:
    """Return what places every line and every sample of one scan.

    Args:
        label: The parsed ISIS label of one scan.

    Returns:
        down: What every line holds, the latitude of it or its northing.
        across: What every sample holds, the longitude of it or its easting.
        polar: The grid the two are measured on, and None for a cylindrical one.

    Raises:
        ValueError: When the label names a projection or a convention this
            cannot read.
    """
    name = label["ProjectionName"]
    if name not in (CYLINDRICAL, POLAR):
        raise ValueError(f"Cannot place a {name} grid.")
    for key, wanted in CONVENTIONS.items():
        if label[key] != wanted:
            raise ValueError(f"Cannot place a grid whose {key} is {label[key]}.")
    radius = float(label["EquatorialRadius"])
    resolution = float(label["PixelResolution"])
    # The corner it starts from, moved in half a pixel so an axis holds centres.
    half = resolution / 2.0
    top = float(label["UpperLeftCornerY"]) - half
    left = float(label["UpperLeftCornerX"]) + half
    lines, samples = np.arange(int(label["Lines"])), np.arange(int(label["Samples"]))
    if name == POLAR:
        return (
            top - lines * resolution,
            left + samples * resolution,
            (
                float(label["CenterLongitude"]),
                float(label["CenterLatitude"]) > 0.0,
                radius,
            ),
        )
    # How many degrees one pixel spans, the same in both directions.
    step = float(np.degrees(resolution / radius))
    return (
        np.degrees(top / radius) - lines * step,
        float(label["CenterLongitude"]) + np.degrees(left / radius) + samples * step,
        None,
    )
