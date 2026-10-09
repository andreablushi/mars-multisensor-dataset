"""What the ground truth reads: ODE's features, the labels written, the verdicts."""

from __future__ import annotations

from dataclasses import asdict

import httpx

from analysis import paths
from analysis.ground_truth.artifacts import LABELS
from analysis.ground_truth.models.feature import Feature
from analysis.ground_truth.models.label import Label
from common.disk import parquet
from common.disk.files import read_json, read_jsonl, write_json, write_jsonl
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
    """Read back every labelled tile, in the order they were written."""
    return parquet.read_rows(Label, LABELS, paths.LABELS_ROOT / paths.LABELS_NAME)


def read_verdicts() -> dict[str, bool]:
    """Read the review's verdict on every tile, from a file the pipeline never writes.

    Returns:
        verdicts: Whether each reviewed tile was accepted, by its name.
    """
    if not paths.VERDICTS_PATH.is_file():
        return {}
    return read_json(paths.VERDICTS_PATH)


def write_verdicts(verdicts: dict[str, bool]) -> None:
    """Write the review's verdict on every tile, sorted by tile."""
    write_json(paths.VERDICTS_PATH, verdicts, indent=1, sort_keys=True)


def refused_tiles(verdicts: dict[str, bool]) -> set[str]:
    """Return the names of the tiles the verdicts refuse."""
    return {tile for tile, accepted in verdicts.items() if not accepted}


def read_refused() -> set[str]:
    """Read the names of the tiles the review refused, none when nothing was."""
    return refused_tiles(read_verdicts())
