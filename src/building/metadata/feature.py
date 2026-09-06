"""What the dataset says about one geological feature, and where it sits on Mars."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from analysis.selector.models.selection import SelectedFeature
from building.models.feature import FeatureFrame
from utils.disk import parquet
from utils.geometry import geodesy


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


def feature_metadata(feature: SelectedFeature) -> FeatureMetadata:
    """Return what the dataset holds about one feature the selection kept.

    Args:
        feature: The feature's own row, as the selection wrote it.

    Returns:
        The metadata, its frame centred on the catalogue box.
    """
    centre_lon, centre_lat = geodesy.bbox_centre(
        feature.min_lat, feature.max_lat, feature.west_lon, feature.east_lon
    )
    span = geodesy.longitude_span(feature.west_lon, feature.east_lon)
    # A degree of longitude is longest at the equator, so the box reaches
    # furthest east at whichever of its latitudes lies nearest to it.
    widest = (
        0.0
        if feature.min_lat <= 0.0 <= feature.max_lat
        else min(abs(feature.min_lat), abs(feature.max_lat))
    )
    return FeatureMetadata(
        frame=FeatureFrame(
            feature_class=feature.feature_class,
            feature_name=feature.feature_name,
            centre_lon=centre_lon,
            centre_lat=centre_lat,
            min_lat=feature.min_lat,
            max_lat=feature.max_lat,
            west_lon=feature.west_lon,
            east_lon=feature.east_lon,
            east_m=geodesy.eastward_m(span / 2.0, widest),
            north_m=geodesy.northward_m((feature.max_lat - feature.min_lat) / 2.0),
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


SCHEMA = parquet.schema_of(FeatureMetadata)
