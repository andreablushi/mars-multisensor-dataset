"""Where one feature sits on Mars, which every position of it is relative to."""

from __future__ import annotations

from dataclasses import dataclass

from utils.geometry import geodesy


@dataclass(frozen=True, slots=True)
class FeatureFrame:
    """The local frame one feature's observations are placed in.

    Attributes:
        feature_class: The feature class, such as Crater.
        feature_name: The feature name as ODE spells it.
        min_lat: The southernmost latitude of the catalogue box, in degrees.
        max_lat: The northernmost latitude of the box, in degrees.
        west_lon: The westernmost longitude of the box, 0 to 360.
        east_lon: The easternmost longitude of the box, 0 to 360.
    """

    feature_class: str
    feature_name: str
    min_lat: float
    max_lat: float
    west_lon: float
    east_lon: float

    @property
    def centre_lon(self) -> float:
        """Return the longitude the local projection is centred on.

        Returns:
            centre: The centre of the catalogue box in degrees, -180 to 180.
        """
        return geodesy.bbox_centre(
            self.min_lat, self.max_lat, self.west_lon, self.east_lon
        )[0]

    @property
    def centre_lat(self) -> float:
        """Return the latitude the local projection is centred on.

        Returns:
            centre: The centre of the catalogue box in degrees.
        """
        return geodesy.bbox_centre(
            self.min_lat, self.max_lat, self.west_lon, self.east_lon
        )[1]
