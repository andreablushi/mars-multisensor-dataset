"""One downloaded observation, as every stage that reads the metadata sees it."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from shared.models.feature import Feature


@dataclass(frozen=True, slots=True)
class Observation:
    """One downloaded observation, as ODE published it.

    Attributes:
        pdsid: The PDS product identifier.
        ihid: The instrument host identifier.
        iid: The instrument identifier.
        pt: The product type.
        start: When the observation started.
        stop: When the observation finished, or None when none was published.
        wkt: The footprint as well-known text, left unparsed.
        map_scale_m: The ground size of one pixel, or None when unpublished.
    """

    pdsid: str
    ihid: str
    iid: str
    pt: str
    start: datetime
    stop: datetime | None
    wkt: str
    map_scale_m: float | None

    @property
    def is_track(self) -> bool:
        """Report whether the footprint is a ground track rather than an area.

        Returns:
            track: True when the footprint carries no polygon and must be buffered.
        """
        return self.wkt.startswith(("LINESTRING", "MULTILINESTRING"))

    @property
    def duration_s(self) -> float:
        """Return how long the observation lasted.

        Returns:
            seconds: The elapsed seconds, or zero when no stop was published.
        """
        return (self.stop - self.start).total_seconds() if self.stop else 0.0


@dataclass(frozen=True, slots=True)
class ObservationSet:
    """One instrument set's observations, and what they belong to.

    Attributes:
        feature: The feature box the records were downloaded for.
        set_key: The instrument set identifier the records were asked for by.
        observations: The set's observations, in chronological order.
        discarded: How many records could not be used.
    """

    feature: Feature
    set_key: str
    observations: list[Observation]
    discarded: int
