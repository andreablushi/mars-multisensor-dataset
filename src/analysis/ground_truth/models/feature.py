"""One named feature of Mars, as ODE publishes it."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Feature:
    """One feature of the IAU nomenclature, bounded by the box ODE gives it.

    Attributes:
        name: The feature's name, such as "Aram Chaos".
        feature_class: The IAU descriptor it is named under, such as "Chaos".
        min_lat: The southernmost planetocentric latitude in degrees.
        max_lat: The northernmost planetocentric latitude in degrees.
        west_lon: The westernmost longitude in degrees, 0 to 360.
        east_lon: The easternmost longitude in degrees, 0 to 360, equal to the
            westernmost where the feature circles a pole.
    """

    name: str
    feature_class: str
    min_lat: float
    max_lat: float
    west_lon: float
    east_lon: float
