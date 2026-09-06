"""Reading the written selection back, which is how the building half reads it."""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

import utils.disk.paths as paths
from analysis.selector.artifacts.write import FEATURES, OBSERVATIONS
from analysis.selector.models.selection import (
    SelectedFeature,
    SelectedObservation,
    Selection,
)


def read_dataset_list(root: Path = paths.SELECTION_ROOT) -> list[Selection]:
    """Read back every feature the selection stage searched, and what each keeps.

    A feature the filter refused is read back too, holding no observation, so a
    caller sees what to leave alone as plainly as what to download.

    Args:
        root: The directory the selection was written in.

    Returns:
        One entry per feature searched, in the order they were written, each
        carrying the observations it keeps, oldest first.

    Raises:
        FileNotFoundError: When no selection has been written there.
    """
    features = root / paths.SELECTED_FEATURES_NAME
    observations = root / paths.SELECTED_OBSERVATIONS_NAME
    if not features.is_file() or not observations.is_file():
        raise FileNotFoundError(f"no selection was written in {root}")
    kept: dict[tuple[str, str], list[SelectedObservation]] = {}
    for row in pq.read_table(observations, schema=OBSERVATIONS).to_pylist():
        observation = SelectedObservation(**row)
        held = kept.setdefault(
            (observation.feature_class, observation.feature_name), []
        )
        held.append(observation)
    return [
        Selection(
            feature=feature,
            observations=kept.get((feature.feature_class, feature.feature_name), []),
        )
        for feature in (
            SelectedFeature(**row)
            for row in pq.read_table(features, schema=FEATURES).to_pylist()
        )
    ]
