"""The evaluation labels written down and read back, and the review's refusals."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from analysis import paths
from analysis.ground_truth.models.label import Label
from common.disk import parquet
from common.disk.files import read_json

LABELS = parquet.schema_of(Label)


def write_labels(labels: Sequence[Label], root: Path = paths.LABELS_ROOT) -> None:
    """Write every labelled tile down, the drawn ones marked so.

    Args:
        labels: Every labelled tile.
        root: The directory the file is written in, made when it is missing.
    """
    parquet.write_rows(labels, LABELS, root / paths.LABELS_NAME)


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
