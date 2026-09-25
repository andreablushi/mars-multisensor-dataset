"""What one instrument set covered of one tile, observation by observation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from analysis.coverage.models.summary import Summary
from analysis.models.instrument import InstrumentSet


@dataclass(frozen=True, slots=True)
class Event:
    """One row of the per-observation coverage record.

    Attributes:
        tile: The tile's name, such as "b123_c0456".
        ihid: The instrument host identifier.
        iid: The instrument identifier.
        pt: The product type.
        pdsid: The PDS product identifier.
        t_start: When the observation started.
        own_km2: Ground this footprint covers inside the tile.
        cum_frac: The share of the tile its instrument set has covered so far.
        width_km: The swath width used, or None when the footprint had area.
        pixels: How many of the instrument's pixels landed inside the tile.
        mask: The tile's cells this footprint fills, packed as a bitmap or a list.
    """

    tile: str
    ihid: str
    iid: str
    pt: str
    pdsid: str
    t_start: datetime
    own_km2: float
    cum_frac: float
    width_km: float | None
    pixels: float
    mask: bytes


@dataclass(frozen=True, slots=True)
class SetCoverage:
    """What one instrument set covered of one tile, read back off disk.

    Attributes:
        events: The set's observations in chronological order.
        summary: The single row describing the set as a whole.
    """

    events: list[Event]
    summary: Summary

    @property
    def label(self) -> str:
        """Return the set's instrument and product type, such as "CTX EDR"."""
        return InstrumentSet.from_key(self.summary.set_key).label

    @property
    def observed(self) -> bool:
        """Report whether the set holds any observation of this tile."""
        return bool(self.events)

    @property
    def reason(self) -> str:
        """Return why the set holds nothing to draw, or nothing when it was observed."""
        return "" if self.observed else "no observations"
