"""What the search left of one tile, as the rows the selection is written as."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True, slots=True)
class SelectedTile:
    """One searched tile, and the window it earned or did not.

    Attributes:
        tile: The tile's name, such as "b123_c0456".
        band: The latitude band it sits in, counted from the south pole.
        column: Its place along that band, counted east from the prime meridian.
        min_lat: The southernmost planetocentric latitude in degrees.
        max_lat: The northernmost planetocentric latitude in degrees.
        west_lon: The westernmost longitude in degrees, 0 to 360.
        east_lon: The easternmost longitude in degrees, 0 to 360.
        kept: Whether the tile earned a place at all.
        area_km2: How much ground it covers.
        start: When the earliest observation it keeps was taken, or None where
            it earned no window.
        end: When the latest one was taken, or None where it earned none.
        days: How long the window runs, in days.
        geo_mean: The insisted shares rooted together, as a share of the tile.
        taken: How many observations it keeps, the standing looks counted in.
    """

    tile: str
    band: int
    column: int
    min_lat: float
    max_lat: float
    west_lon: float
    east_lon: float
    kept: bool
    area_km2: float
    start: datetime | None
    end: datetime | None
    days: float
    geo_mean: float
    taken: int


@dataclass(frozen=True, slots=True)
class SelectedObservation:
    """One observation a tile keeps, named as its archive names it.

    Attributes:
        tile: The tile it was kept for.
        ihid: The instrument host identifier.
        iid: The instrument identifier.
        pt: The product type.
        pdsid: The PDS product identifier, which is what a download asks for.
        t_start: When the observation started.
        standing: Whether it was kept from outside the window, which only a
            timeless instrument can be.
    """

    tile: str
    ihid: str
    iid: str
    pt: str
    pdsid: str
    t_start: datetime
    standing: bool


@dataclass(frozen=True, slots=True)
class Selection:
    """What one tile contributes to the written selection.

    Attributes:
        tile: The tile's own row, whether or not it earned a window.
        observations: The observations it keeps, oldest first, and nothing at
            all where it earned no window.
    """

    tile: SelectedTile
    observations: list[SelectedObservation] = field(default_factory=list)
