"""Writing the selection down: which features earned a place, and what to download."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import utils.disk.paths as paths
from analysis.selector.models.selection import (
    SelectedFeature,
    SelectedObservation,
    Selection,
)
from utils.disk import parquet

FEATURES = parquet.schema_of(SelectedFeature)
OBSERVATIONS = parquet.schema_of(SelectedObservation)


def write_selection(
    picked: Sequence[Selection], root: Path = paths.SELECTION_ROOT
) -> tuple[Path, Path]:
    """Write every searched feature down, and every observation they keep.

    Args:
        picked: What the search left of each feature, in the order to write them.
        root: The directory the two files are written in, made when it is missing.

    Returns:
        The features file and the observations file, in that order.
    """
    features = root / paths.SELECTED_FEATURES_NAME
    observations = root / paths.SELECTED_OBSERVATIONS_NAME
    parquet.write([one.feature for one in picked], FEATURES, features)
    parquet.write(
        [kept for one in picked for kept in one.observations],
        OBSERVATIONS,
        observations,
    )
    return features, observations
