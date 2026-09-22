"""One tile's ice at two depths and its radar surface power, as SWIM maps them."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SwimSettings:
    """The settled choices for sampling the SWIM maps under every tile.

    Attributes:
        threshold_1_5m: The consistency from which a point is icy between 1 and 5 m.
        threshold_5m: The consistency from which a point is icy below 5 m.
        icy_share: The share of its points icy at a depth from which a tile is.
    """

    threshold_1_5m: float
    threshold_5m: float
    icy_share: float


@dataclass(frozen=True, slots=True)
class IceLabel:
    """One tile, the ice SWIM finds under its grid at two depths, and its radar.

    Attributes:
        tile: The tile's name, such as "b123_c0456".
        centre_lon: The longitude its grid is centred on, -180 to 180.
        centre_lat: The latitude its grid is centred on, in degrees.
        consistency_1_5m: The ice consistency between 1 and 5 m at every point,
            rows from the north and points from the west. (P * P,)
        consistency_5m: The same below 5 m. (P * P,)
        radar_surface_power: The ice consistency of SHARAD's surface echo power
            alone at every point, one of -1, -0.5, 0, 0.5 and 1. (P * P,)
        icy_points_1_5m: Whether each point is icy between 1 and 5 m. (P * P,)
        icy_points_5m: Whether each point is icy below 5 m. (P * P,)
        ice_1_5m: Whether enough of its points are icy between 1 and 5 m.
        ice_5m: Whether enough of its points are icy below 5 m.
    """

    tile: str
    centre_lon: float
    centre_lat: float
    consistency_1_5m: tuple[float, ...]
    consistency_5m: tuple[float, ...]
    radar_surface_power: tuple[float, ...]
    icy_points_1_5m: tuple[bool, ...]
    icy_points_5m: tuple[bool, ...]
    ice_1_5m: bool
    ice_5m: bool
