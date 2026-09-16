"""The tile both halves are keyed by, and the box it is bounded to."""

from __future__ import annotations

from dataclasses import dataclass

from shared.maths import geodesy


@dataclass(frozen=True, slots=True)
class Tile:
    """One tile of the grid Mars is split into, with its bounding box.

    Attributes:
        band: The latitude band it sits in, counted from the south pole.
        column: Its place along that band, counted east from the prime meridian.
        min_lat: The southernmost planetocentric latitude in degrees.
        max_lat: The northernmost planetocentric latitude in degrees.
        west_lon: The westernmost longitude in degrees, 0 to 360.
        east_lon: The easternmost longitude in degrees, 0 to 360, equal to the
            westernmost where the tile circles a pole.
    """

    band: int
    column: int
    min_lat: float
    max_lat: float
    west_lon: float
    east_lon: float

    @property
    def band_name(self) -> str:
        """Return the band as every path spells it.

        Returns:
            name: The band, such as "b123".
        """
        return f"b{self.band:03d}"

    @property
    def column_name(self) -> str:
        """Return the column as every path spells it.

        Returns:
            name: The column, such as "c0456".
        """
        return f"c{self.column:04d}"

    @property
    def name(self) -> str:
        """Return what tells this tile from every other.

        Returns:
            name: The band and the column, such as "b123_c0456".
        """
        return f"{self.band_name}_{self.column_name}"

    @property
    def centre_lon(self) -> float:
        """Return the longitude the local projection is centred on.

        Returns:
            centre: The centre of the box in degrees, -180 to 180.
        """
        return geodesy.bbox_centre(
            self.min_lat, self.max_lat, self.west_lon, self.east_lon
        )[0]

    @property
    def centre_lat(self) -> float:
        """Return the latitude the local projection is centred on.

        Returns:
            centre: The centre of the box in degrees.
        """
        return geodesy.bbox_centre(
            self.min_lat, self.max_lat, self.west_lon, self.east_lon
        )[1]

    @property
    def circles_a_pole(self) -> bool:
        """Return whether the tile runs through every longitude.

        Returns:
            circles: True when the west and east longitudes are the same.
        """
        return self.west_lon == self.east_lon
