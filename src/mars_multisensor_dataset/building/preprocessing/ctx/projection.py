"""The grid ISIS projects one CTX scan onto: the map it takes, the label it writes."""

from __future__ import annotations

import numpy as np

from mars_multisensor_dataset.building.configs import ctx as configs
from mars_multisensor_dataset.building.preprocessing.common.models.position import (
    Position,
)
from mars_multisensor_dataset.common.maths.geodesy import HALF_TURN, POLE, PolarGrid

# The two projections a scan is written in, the second on a polar tile.
EQUATORIAL = "Equirectangular"
POLAR = "PolarStereographic"

# The only reading of latitude and longitude this places; the domain does not matter.
CONVENTIONS = {
    "LatitudeType": "Planetocentric",
    "LongitudeDirection": "PositiveEast",
}


def map_template(grid: PolarGrid | None, latitude: float) -> str:
    """Return the ISIS map that projects a scan onto one tile's grid.

    Args:
        grid: The polar grid of the tile, or None for an equatorial one.
        latitude: The tile's centre latitude, where an equatorial pixel is square on
            the ground.

    Returns:
        template: The map template, as cam2map reads it.
    """
    if grid is None:
        return configs.MAP.format(
            name=EQUATORIAL, latitude=latitude, longitude=HALF_TURN
        )
    return configs.MAP.format(
        name=POLAR, latitude=POLE if grid.north else -POLE, longitude=grid.centre_lon
    )


def grid_position(label: dict[str, str]) -> Position:
    """Return where every line and every sample of one projected scan sits.

    Args:
        label: The parsed ISIS label of one scan.

    Returns:
        position: The latitude or northing of every line, the longitude or easting of
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
        return Position(
            top - lines * resolution,
            left + samples * resolution,
            True,
            PolarGrid(
                float(label["CenterLongitude"]),
                float(label["CenterLatitude"]) > 0.0,
                radius,
            ),
        )
    scale = float(label["Scale"])
    across = scale * np.cos(np.radians(float(label["CenterLatitude"])))
    return Position(
        (top / resolution - lines) / scale,
        float(label["CenterLongitude"]) + (left / resolution + samples) / across,
        True,
    )
