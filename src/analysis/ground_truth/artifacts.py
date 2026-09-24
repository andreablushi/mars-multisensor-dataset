"""The evaluation labels written down and read back, and the review's refusals."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pyarrow.parquet as pq

from analysis import paths
from analysis.ground_truth.models.label import Label
from common.disk import parquet

LABELS = parquet.schema_of(Label)


def write_labels(labels: Sequence[Label], root: Path = paths.LABELS_ROOT) -> Path:
    """Write every labelled tile down, the drawn ones marked so.

    Args:
        labels: Every labelled tile.
        root: The directory the file is written in, made when it is missing.

    Returns:
        path: The file the labels were written to.
    """
    path = root / paths.LABELS_NAME
    parquet.write(labels, LABELS, path)
    return path


def read_labels(root: Path = paths.LABELS_ROOT) -> list[Label]:
    """Read back every labelled tile.

    Args:
        root: The directory the labels were written in.

    Returns:
        labels: Every labelled tile, in the order they were written.

    Raises:
        FileNotFoundError: When no labels have been written there.
    """
    path = root / paths.LABELS_NAME
    if not path.is_file():
        raise FileNotFoundError(f"no evaluation labels were written in {root}")
    return [Label(**row) for row in pq.read_table(path, schema=LABELS).to_pylist()]


def read_refused(path: Path = paths.VERDICTS_PATH) -> set[str]:
    """Read the tiles the review refused, from a file the pipeline never writes.

    Args:
        path: The review's verdicts, each tile mapped to whether it was accepted.

    Returns:
        refused: The names of the refused tiles, none when nothing was reviewed.
    """
    if not path.is_file():
        return set()
    verdicts = json.loads(path.read_text(encoding="utf-8"))
    return {tile for tile, accepted in verdicts.items() if not accepted}
