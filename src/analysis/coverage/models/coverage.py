"""What one instrument set covered of one feature, observation by observation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from analysis.coverage.models.summary import Summary
from analysis.models.instrument import InstrumentSet


@dataclass(frozen=True, slots=True)
class Event:
    """One row of the per-observation coverage record.

    Attributes:
        feature_class: The feature class, such as Crater or Collis.
        feature_name: The feature name as ODE spells it.
        ihid: The instrument host identifier.
        iid: The instrument identifier.
        pt: The product type.
        pdsid: The PDS product identifier.
        t_start: When the observation started.
        t_stop: When the observation finished, or None when none was published.
        own_km2: Ground this footprint covers inside the feature.
        new_km2: Ground its instrument set had not covered before.
        cum_km2: Ground its instrument set has covered including this one.
        cum_frac: The same as a share of the feature.
        width_km: The swath width used, or None when the footprint had area.
        pixels: How many of the instrument's pixels landed inside the feature.
        mask: The feature's cells this footprint fills, packed as a bitmap or a list.
    """

    feature_class: str
    feature_name: str
    ihid: str
    iid: str
    pt: str
    pdsid: str
    t_start: datetime
    t_stop: datetime | None
    own_km2: float
    new_km2: float
    cum_km2: float
    cum_frac: float
    width_km: float | None
    pixels: float
    mask: bytes


@dataclass(frozen=True, slots=True)
class SetCoverage:
    """What one instrument set covered of one feature, read back off disk.

    Attributes:
        events: The set's observations in chronological order.
        summary: The single row describing the set as a whole.
        pending: Whether the set has records downloaded but never measured.
    """

    events: list[Event]
    summary: Summary
    pending: bool = False

    @property
    def label(self) -> str:
        """Return the short readable name for the instrument set.

        Returns:
            label: The instrument and product type, such as "CTX EDR", with any pattern.
        """
        return InstrumentSet.from_key(self.summary.set_key).label

    @property
    def observed(self) -> bool:
        """Report whether the set holds any observation of this feature.

        Returns:
            observed: True when the set has at least one observation.
        """
        return bool(self.events)

    @property
    def reason(self) -> str:
        """Return why the set holds nothing to draw.

        Returns:
            reason: What is missing, or an empty string when the set was observed.
        """
        if self.observed:
            return ""
        return "downloaded, not yet measured" if self.pending else "no observations"
