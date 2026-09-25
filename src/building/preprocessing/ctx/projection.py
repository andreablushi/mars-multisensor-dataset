"""The grid ISIS projects one CTX scan onto: the map it takes, the label it writes."""

from __future__ import annotations

import numpy as np

from building.configs import ctx as configs
from building.preprocessing.common.models.samples import Samples
from common.maths.geodesy import PolarGrid

# The two projections a scan is written in, the second on a polar tile.
EQUATORIAL = "SimpleCylindrical"
POLAR = "PolarStereographic"

# The only reading of latitude and longitude this places; the domain does not matter.
CONVENTIONS = {
    "LatitudeType": "Planetocentric",
    "LongitudeDirection": "PositiveEast",
}


def map_template(grid: PolarGrid | None) -> str:
    """Return the ISIS map that projects a scan onto one tile's grid.

    Args:
        grid: The polar grid of the tile, or None for an equatorial one.

    Returns:
        template: The map template, as cam2map reads it.
    """
    if grid is None:
        return configs.MAP.format(name=EQUATORIAL, latitude=0.0, longitude=180.0)
    return configs.MAP.format(
        name=POLAR, latitude=90.0 if grid[1] else -90.0, longitude=grid[0]
    )


def grid_samples(label: dict[str, str]) -> Samples:
    """Return where every line and every sample of one projected scan sits.

    Args:
        label: The parsed ISIS label of one scan.

    Returns:
        samples: The latitude or northing of every line, the longitude or easting of
            every sample, and the polar grid they are measured on, if any.

    Raises:
        ValueError: When the projection or convention cannot be read.
    """
    name = label["ProjectionName"]
    if name not in (EQUATORIAL, POLAR):
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
        return Samples(
            top - lines * resolution,
            left + samples * resolution,
            True,
            (
                float(label["CenterLongitude"]),
                float(label["CenterLatitude"]) > 0.0,
                radius,
            ),
        )
    # How many degrees one pixel spans, the same in both directions.
    step = float(np.degrees(resolution / radius))
    return Samples(
        np.degrees(top / radius) - lines * step,
        float(label["CenterLongitude"]) + np.degrees(left / radius) + samples * step,
        True,
        None,
    )
