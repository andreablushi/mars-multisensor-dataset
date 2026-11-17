"""The lava flow labels as written, and the review's verdict on each crop."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from mars_multisensor_dataset.analysis import paths
from mars_multisensor_dataset.analysis.ground_truth.lavaflow.models import (
    LavaFlowLabel,
)
from mars_multisensor_dataset.common.disk import parquet
from mars_multisensor_dataset.common.disk.files import read_json, write_json

LAVAFLOW_LABELS = parquet.schema_of(LavaFlowLabel)


def read_lavaflow_labels(root: Path = paths.LABELS_ROOT) -> list[LavaFlowLabel]:
    """Read back every candidate crop, in the order they were written."""
    return parquet.read_rows(
        LavaFlowLabel, LAVAFLOW_LABELS, root / paths.LAVAFLOW_LABELS_NAME
    )


def write_lavaflow_labels(
    labels: Sequence[LavaFlowLabel], root: Path = paths.LABELS_ROOT
) -> None:
    """Write every candidate crop down, each with the verdict it holds."""
    parquet.write_rows(labels, LAVAFLOW_LABELS, root / paths.LAVAFLOW_LABELS_NAME)


def read_lavaflow_verdicts() -> dict[str, str]:
    """Read the review's verdict on every crop, from a file the pipeline never writes.

    Returns:
        verdicts: What the review saw in each reviewed crop, by tile and track.
    """
    if not paths.LAVAFLOW_VERDICTS_PATH.is_file():
        return {}
    return read_json(paths.LAVAFLOW_VERDICTS_PATH)


def write_lavaflow_verdicts(verdicts: dict[str, str]) -> None:
    """Write the review's verdict on every crop, sorted by tile and track."""
    write_json(paths.LAVAFLOW_VERDICTS_PATH, verdicts, indent=1, sort_keys=True)


def judged_labels(
    labels: Sequence[LavaFlowLabel], verdicts: dict[str, str]
) -> list[LavaFlowLabel]:
    """Return every candidate crop holding the verdict the review gave it, if any."""
    return [replace(one, verdict=verdicts.get(one.crop)) for one in labels]
