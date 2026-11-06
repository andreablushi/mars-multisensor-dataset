"""The ice labels as written: one row per tile the SWIM maps cover."""

from __future__ import annotations

from collections.abc import Sequence

from analysis import paths
from analysis.ground_truth.swim.models import IceLabel
from common.disk import parquet

ICE_LABELS = parquet.schema_of(IceLabel)


def read_ice_labels() -> list[IceLabel]:
    """Read back every tile the SWIM maps labelled, in the order they were written."""
    return parquet.read_rows(IceLabel, ICE_LABELS, paths.ICE_LABELS_PATH)


def write_ice_labels(labels: Sequence[IceLabel]) -> None:
    """Write every tile the SWIM maps labelled down."""
    parquet.write_rows(labels, ICE_LABELS, paths.ICE_LABELS_PATH)
