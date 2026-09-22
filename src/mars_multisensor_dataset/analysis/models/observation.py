"""One downloaded observation, as every stage that reads the metadata sees it."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class Observation:
    """One downloaded observation, as ODE published it.

    Attributes:
        pdsid: The PDS product identifier.
        ihid: The instrument host identifier.
        iid: The instrument identifier.
        pt: The product type.
        start: When the observation started.
        wkt: The footprint as well-known text, left unparsed.
        north_wkt: The footprint in north polar stereographic metres, or None.
        south_wkt: The same in south polar stereographic metres.
    """

    pdsid: str
    ihid: str
    iid: str
    pt: str
    start: datetime
    wkt: str
    north_wkt: str | None
    south_wkt: str | None

    @property
    def is_track(self) -> bool:
        """Return whether the footprint is a ground track, buffered into an area."""
        return self.wkt.startswith(("LINESTRING", "MULTILINESTRING"))


@dataclass(frozen=True, slots=True)
class ObservationSet:
    """One instrument set's observations over one group.

    Attributes:
        set_key: The instrument set identifier the records were asked for by.
        observations: The set's observations, in chronological order.
        discarded: How many records could not be used.
    """

    set_key: str
    observations: list[Observation]
    discarded: int
