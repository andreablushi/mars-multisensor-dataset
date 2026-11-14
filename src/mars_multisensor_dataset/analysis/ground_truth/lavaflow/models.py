"""One SHARAD crop a lava flow reflector is looked for in, and what settles them."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from mars_multisensor_dataset.common.maths.box import Boxed

REFLECTOR = "reflector"
NONE = "none"
UNSURE = "unsure"
VERDICTS = (REFLECTOR, NONE, UNSURE)


@dataclass(slots=True)
class LavaFlowRegion(Boxed):
    """One region a paper reports reflectors in, and the tracks it names there.

    Attributes:
        source: The paper the region and its tracks are read from.
        min_lat: The southernmost latitude of the region in degrees.
        max_lat: The northernmost latitude of the region in degrees.
        west_lon: The westernmost longitude of the region in degrees, 0 to 360.
        east_lon: The easternmost longitude of the region in degrees, 0 to 360.
        tracks: The tracks the paper names, as ODE names them, such as "s_00495001".
    """

    source: str
    min_lat: float
    max_lat: float
    west_lon: float
    east_lon: float
    tracks: list[str]


@dataclass(slots=True)
class LavaFlowSettings:
    """The settled choices for listing the crops a reflector is looked for in.

    Attributes:
        unnamed_per_tile: How many tracks no paper names are kept over every tile.
        regions: Every region a paper reports reflectors in, by its name.
    """

    unnamed_per_tile: int
    regions: dict[str, LavaFlowRegion]


@dataclass(frozen=True, slots=True)
class LavaFlowLabel:
    """One track over one tile of a region, and what the review saw in it.

    Attributes:
        tile: The tile's name, such as "b123_c0456".
        track: The track, as ODE names it, such as "s_00495001".
        pdsid: The radargram's PDS product identifier.
        t_start: When the track started.
        region: The region it lies in, as the config names it.
        source: The paper the region is read from.
        named: Whether that paper names the track.
        verdict: What the review saw, one of `VERDICTS`, or None before it.
    """

    tile: str
    track: str
    pdsid: str
    t_start: datetime
    region: str
    source: str
    named: bool
    verdict: str | None = None

    @property
    def crop(self) -> str:
        """Return what tells this crop from every other, as the verdicts key it."""
        return f"{self.tile}/{self.track}"
