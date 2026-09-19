"""Fetching the ODE feature catalogue, the one source every label is read from."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from analysis import paths
from analysis.labels.models.feature import Feature
from analysis.metadata.ode import ODEClient
from common.disk.files import read_jsonl, write_jsonl
from common.fetch.ode import ODE_TARGET


def read_features(
    path: Path = paths.FEATURES_PATH, *, refresh: bool = False
) -> list[Feature]:
    """Read the Mars feature catalogue, fetching it from ODE when none is cached.

    Args:
        path: Where the catalogue is cached.
        refresh: Whether to fetch it again even when it is cached.

    Returns:
        features: Every feature ODE publishes, each once.

    Raises:
        KeyError: When ODE answers without the catalogue it always publishes.
    """
    if path.exists() and not refresh:
        return [Feature(**row) for row in read_jsonl(path)]
    with ODEClient() as client:
        results = client.query({"query": "featuredata", "odemetadb": ODE_TARGET})
    # ODE publishes some features twice, so the first of each is kept
    features = list(
        dict.fromkeys(
            Feature(
                name=item["FeatureName"],
                feature_class=item["FeatureClass"],
                min_lat=float(item["MinLat"]),
                max_lat=float(item["MaxLat"]),
                west_lon=float(item["WestLon"]),
                east_lon=float(item["EastLon"]),
            )
            for item in results["Features"]["Feature"]
        )
    )
    write_jsonl(path, [asdict(one) for one in features])
    return features
