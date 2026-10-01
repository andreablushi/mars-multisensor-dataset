"""The evaluation labels written down."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from analysis import paths
from analysis.ground_truth.models.label import Label
from common.disk import parquet

LABELS = parquet.schema_of(Label)


def write_labels(labels: Sequence[Label], root: Path = paths.LABELS_ROOT) -> None:
    """Write every labelled tile down, the drawn ones marked so.

    Args:
        labels: Every labelled tile.
        root: The directory the file is written in, made when it is missing.
    """
    parquet.write_rows(labels, LABELS, root / paths.LABELS_NAME)
