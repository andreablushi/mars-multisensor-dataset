"""Placing one CTX scan on the grid its label projects it onto."""

from __future__ import annotations

import numpy as np

from building.preprocessing.common.models.relative_position import PolarGrid

# The two projections ASU writes a CTX RDR in, the second above about seventy
# degrees, where a cylindrical grid stops holding a scan in any usable shape.
CYLINDRICAL = "SimpleCylindrical"
POLAR = "PolarStereographic"

# The only reading of latitude and longitude this places. Which domain a label
# numbers its longitudes in is not among them, since every longitude read here
# is wrapped before it is compared and none is ever compared as a number.
CONVENTIONS = {
    "LatitudeType": "Planetocentric",
    "LongitudeDirection": "PositiveEast",
}


def load(
    label: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, PolarGrid | None]:
    """Return what places every line and every sample of one scan.

    Args:
        label: The parsed ISIS label of one scan.

    Returns:
        What every line holds and what every sample holds, and the polar grid
        the two are measured on. A cylindrical grid gives the latitude of every
        line, falling southward, and the longitude of every sample, rising
        eastward, both in degrees, and no grid beside them. A polar one gives
        the northing and the easting in the projection's own metres, and the
        grid that turns them back into degrees.

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
    # The corner the projection starts from, moved in by half a pixel so that
    # every axis holds the centre of what it places rather than its edge.
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
