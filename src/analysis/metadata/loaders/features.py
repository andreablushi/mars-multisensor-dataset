"""Loading the ODE feature catalog, from the cache or from ODE itself."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import utils.disk.paths as paths
from analysis.metadata.fetchers.features import fetch_features
from analysis.metadata.ode import ODEClient
from analysis.models.feature import Feature
from utils.disk.files import read_jsonl, write_jsonl
from utils.disk.paths import features_path


def load_features(
    client: ODEClient | None = None,
    cache_dir: Path = paths.CATALOG_ROOT,
    *,
    refresh: bool = False,
) -> list[Feature]:
    """Load the geological feature catalog, fetching and caching it when asked.

    Args:
        client: The ODE client to fetch with, or None to read the cache alone.
        cache_dir: Directory holding the cached catalog files.
        refresh: When True, re-fetch and overwrite the cache.

    Returns:
        The list of features.

    Raises:
        FileNotFoundError: When nothing is cached and no client was given to fetch with.
    """
    path = features_path(cache_dir)
    if path.exists() and not refresh:
        return [Feature(**row) for row in read_jsonl(path)]
    if client is None:
        raise FileNotFoundError(f"no feature catalog is cached at {path}")
    features = fetch_features(client)
    write_jsonl(path, [asdict(feature) for feature in features])
    return features
