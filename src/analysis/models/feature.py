"""The geological feature both stages are keyed by."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Feature:
    """A named geological feature with its bounding box.

    Attributes:
        name: The feature name as ODE spells it, for example "Gale".
        feature_class: The feature class, for example "Crater".
        min_lat: The southernmost planetocentric latitude in degrees.
        max_lat: The northernmost planetocentric latitude in degrees.
        west_lon: The westernmost longitude in degrees, 0 to 360.
        east_lon: The easternmost longitude in degrees, 0 to 360.
    """

    name: str
    feature_class: str
    min_lat: float
    max_lat: float
    west_lon: float
    east_lon: float

    @property
    def has_latitude_extent(self) -> bool:
        """Return whether the catalogue gives the feature a latitude span.

        Returns:
            spans: True when the maximum latitude is strictly the larger.
        """
        return self.max_lat > self.min_lat

    @property
    def has_longitude_extent(self) -> bool:
        """Return whether the catalogue bounds the feature in longitude.

        Returns:
            spans: True when the west and east longitudes differ.
        """
        return self.west_lon != self.east_lon

    @property
    def is_point(self) -> bool:
        """Return whether the catalogue gives the feature no extent at all.

        Returns:
            point: True when the feature has neither a latitude nor a longitude span.
        """
        return not self.has_latitude_extent and not self.has_longitude_extent

    @property
    def circles_a_pole(self) -> bool:
        """Return whether the feature runs through every longitude.

        Returns:
            circles: True when the feature has a latitude span but no longitude one.
        """
        return self.has_latitude_extent and not self.has_longitude_extent
