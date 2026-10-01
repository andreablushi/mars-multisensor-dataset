"""One tile the evaluation set labels, the feature it was labelled from, and its box."""

from __future__ import annotations

from dataclasses import dataclass

from common.maths.box import Boxed


@dataclass(frozen=True, slots=True)
class Label(Boxed):
    """One kept tile, and the class the feature catalogue gives it.

    Attributes:
        tile: The tile's name, such as "b123_c0456".
        label: The class it earned, such as "chaos".
        feature: The feature it earned it from, such as "Aram Chaos".
        overlaps: How many features of another class or excluded descriptor touch it.
        centre_offset: How far it sits from its feature's centre, 0 at the centre and 1
            at the edge.
        min_lat: The southernmost latitude of the box its crop is cut to, in degrees.
        max_lat: The northernmost latitude of that box in degrees.
        west_lon: The westernmost longitude of that box in degrees, 0 to 360.
        east_lon: The easternmost longitude of that box, 0 to 360, the westernmost
            at a pole.
        drawn: Whether the balanced draw took it into the evaluation set.
    """

    tile: str
    label: str
    feature: str
    overlaps: int
    centre_offset: float
    min_lat: float
    max_lat: float
    west_lon: float
    east_lon: float
    drawn: bool = False
