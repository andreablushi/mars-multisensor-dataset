"""One observation's ground, and the set of it a coverage computation is handed."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from shapely.geometry.base import BaseGeometry

from analysis.coverage.models.region import FeatureRegion
from shared.models.feature import Feature


@dataclass(frozen=True, slots=True)
class ProjectedObservation:
    """One observation's ground, ready to be folded into a union.

    Attributes:
        pdsid: The PDS product identifier.
        ihid: The instrument host identifier.
        iid: The instrument identifier.
        pt: The product type.
        start: When the observation started.
        stop: When the observation finished, or None when none was published.
        shape: The projected footprint, clipped to the feature.
        width_km: The swath width used, or None when the footprint had area.
        pixel_km2: The ground one of its pixels covers.
    """

    pdsid: str
    ihid: str
    iid: str
    pt: str
    start: datetime
    stop: datetime | None
    shape: BaseGeometry
    width_km: float | None
    pixel_km2: float


@dataclass(frozen=True, slots=True)
class ProjectedSet:
    """One instrument set's ground on one feature, ready to be measured.

    Attributes:
        feature: The feature the footprints were cut to.
        set_key: The instrument set identifier the records were asked for by.
        region: That feature projected into equal-area metres.
        observations: The observations that landed on it, in chronological order.
        discarded: How many stored records could not be measured.
    """

    feature: Feature
    set_key: str
    region: FeatureRegion
    observations: list[ProjectedObservation]
    discarded: int
