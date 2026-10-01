"""What the ground truth reads: ODE's features, the labels written, the refusals."""

from __future__ import annotations

from dataclasses import asdict

import httpx

from analysis import paths
from analysis.ground_truth.artifacts import LABELS
from analysis.ground_truth.models.feature import Feature
from analysis.ground_truth.models.label import Label
from common.disk import parquet
from common.disk.files import read_json, read_jsonl, write_jsonl
from common.fetch.http import TLS_CONTEXT
from common.fetch.ode import ODE_TARGET, fetch_results


def read_features(refresh: bool) -> list[Feature]:
    """Read the Mars feature catalogue, fetching it from ODE when none is cached.

    Args:
        refresh: Whether to fetch it again even when it is cached.

    Returns:
        features: Every feature ODE publishes, each once.

    Raises:
        KeyError: When ODE answers without the catalogue it always publishes.
    """
    if paths.FEATURES_PATH.exists() and not refresh:
        return [Feature(**row) for row in read_jsonl(paths.FEATURES_PATH)]
    with httpx.Client(verify=TLS_CONTEXT) as client:
        results = fetch_results(
            {"query": "featuredata", "odemetadb": ODE_TARGET}, client
        )
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
    write_jsonl(paths.FEATURES_PATH, [asdict(feature) for feature in features])
    return features


def read_labels() -> list[Label]:
    """Read back every labelled tile.

    Returns:
        labels: Every labelled tile, in the order they were written.

    Raises:
        FileNotFoundError: When no labels have been written.
    """
    return parquet.read_rows(Label, LABELS, paths.LABELS_ROOT / paths.LABELS_NAME)


def read_refused() -> set[str]:
    """Read the tiles the review refused, from a file the pipeline never writes.

    Returns:
        refused: The names of the refused tiles, none when nothing was reviewed.
    """
    if not paths.VERDICTS_PATH.is_file():
        return set()
    verdicts = read_json(paths.VERDICTS_PATH)
    return {tile for tile, accepted in verdicts.items() if not accepted}
