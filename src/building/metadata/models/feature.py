"""Where one feature sits on Mars, which every position of it is relative to."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FeatureFrame:
    """The local frame one feature's observations are placed in.

    This is the only place a feature's absolute position is written down. Every
    array the building half stores holds offsets from this frame and nothing
    else, so an array says how it sits on its feature and never where that
    feature is on Mars.

    Attributes:
        feature_class: The feature class, such as Crater.
        feature_name: The feature name as ODE spells it.
        centre_lon: The longitude the local projection is centred on, -180 to 180.
        centre_lat: The latitude it is centred on.
        min_lat: The southernmost latitude of the catalogue box, in degrees.
        max_lat: The northernmost latitude of the box, in degrees.
        west_lon: The westernmost longitude of the box, 0 to 360.
        east_lon: The easternmost longitude of the box, 0 to 360.
        east_m: How far east of the centre the box reaches, in metres.
        north_m: How far north of the centre it reaches, in metres.
    """

    feature_class: str
    feature_name: str
    centre_lon: float
    centre_lat: float
    min_lat: float
    max_lat: float
    west_lon: float
    east_lon: float
    east_m: float
    north_m: float
