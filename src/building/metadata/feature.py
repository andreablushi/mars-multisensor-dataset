"""What the dataset says about one geological feature, and where it sits on Mars."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from analysis.selector.models.selection import SelectedFeature
from building.models.feature import FeatureFrame
from shared.disk import parquet


@dataclass(frozen=True, slots=True)
class FeatureMetadata:
    """One feature of the dataset: what it is, when it was seen, and where it lies.

    Attributes:
        frame: The local frame every observation of it is placed against, which
            is the only place its absolute position is written down.
        area_km2: How much ground the catalogue box covers.
        kept: Whether the filter gave the feature a place at all.
        window_start: When the earliest observation it keeps was taken, or None
            where it earned no window.
        window_end: When the latest one was taken, or None for the same reason.
        window_days: How long that window runs.
        window_share: The insisted shares rooted together, as a share of the
            feature.
        observations_kept: How many observations the filter left it.
    """

    frame: FeatureFrame
    area_km2: float
    kept: bool
    window_start: datetime | None
    window_end: datetime | None
    window_days: float
    window_share: float
    observations_kept: int

    @property
    def identity(self) -> tuple[str, str]:
        """Return what tells this feature from every other.

        Returns:
            identity: Its class and its name.
        """
        return (self.frame.feature_class, self.frame.feature_name)


def feature_metadata(feature: SelectedFeature) -> FeatureMetadata:
    """Return what the dataset holds about one feature the selection kept.

    Args:
        feature: The feature's own row, as the selection wrote it.

    Returns:
        metadata: The metadata, its frame carrying the catalogue box.
    """
    return FeatureMetadata(
        frame=FeatureFrame(
            feature_class=feature.feature_class,
            feature_name=feature.feature_name,
            min_lat=feature.min_lat,
            max_lat=feature.max_lat,
            west_lon=feature.west_lon,
            east_lon=feature.east_lon,
        ),
        area_km2=feature.area_km2,
        kept=feature.kept,
        window_start=feature.start,
        window_end=feature.end,
        window_days=feature.days,
        window_share=feature.geo_mean,
        observations_kept=feature.taken,
    )


SCHEMA = parquet.schema_of(FeatureMetadata)
