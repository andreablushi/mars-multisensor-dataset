"""Building one feature's local frame from the box the catalogue gives it."""

from __future__ import annotations

from building.metadata.models.feature import FeatureFrame
from utils.geometry import geodesy


def feature_frame(feature) -> FeatureFrame:
    """Return the local frame one feature's observations are placed in.

    Args:
        feature: The catalogued feature, carrying the box ODE gives it.

    Returns:
        The frame, centred on the box and carrying how far the box reaches
        from that centre.
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
    return FeatureFrame(
        feature_class=feature.feature_class,
        feature_name=feature.name,
        centre_lon=centre_lon,
        centre_lat=centre_lat,
        min_lat=feature.min_lat,
        max_lat=feature.max_lat,
        west_lon=feature.west_lon,
        east_lon=feature.east_lon,
        east_m=geodesy.eastward_m(span / 2.0, widest),
        north_m=geodesy.northward_m((feature.max_lat - feature.min_lat) / 2.0),
    )
