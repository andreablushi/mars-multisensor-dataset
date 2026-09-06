"""Fetching the ODE feature catalog."""

from __future__ import annotations

from analysis.metadata.ode import ODEClient
from analysis.models.feature import Feature

ODE_META_DB = "mars"


def fetch_features(client: ODEClient) -> list[Feature]:
    """Fetch the full Mars feature catalog from ODE, deduplicated.

    Args:
        client: The ODE client to query with.

    Returns:
        The list of unique features.

    Raises:
        KeyError: When ODE answers without the catalog it always publishes.
    """
    results = client.query({"query": "featuredata", "odemetadb": ODE_META_DB})
    features: list[Feature] = []
    # ODE publishes one feature twice; the first spelling of each is kept
    seen: set[Feature] = set()
    for item in results["Features"]["Feature"]:
        feature = Feature(
            name=item["FeatureName"],
            feature_class=item["FeatureClass"],
            min_lat=float(item["MinLat"]),
            max_lat=float(item["MaxLat"]),
            west_lon=float(item["WestLon"]),
            east_lon=float(item["EastLon"]),
        )
        if feature in seen:
            continue
        seen.add(feature)
        features.append(feature)
    return features
