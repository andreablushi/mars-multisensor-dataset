"""The tables the evaluation notebook writes: its classes, and what their crops hold."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import ipywidgets as widgets
import numpy as np

from analysis.labels.models.label import Label
from analysis.labels.models.settings import Settings
from analysis.visualization.common import tables
from analysis.visualization.common.models.tables import Row
from building.metadata.observation import ObservationMetadata

_CLASSES = (
    "Class",
    "Read from",
    "Tiles labelled",
    "Tiles drawn",
    "Features drawn from",
)
_CROPS = ("Instrument", "Class", "Crops", "Mean value", "Mean spread")


def classes(labels: Sequence[Label], settings: Settings) -> widgets.Widget:
    """Tabulate every class: what it is read from, and how many tiles it holds."""
    held = Counter(one.label for one in labels)
    rows: list[Row] = []
    for name, rule in settings.classes.items():
        drawn = Counter(
            one.feature for one in labels if one.drawn and one.label == name
        )
        read_from = rule.descriptor or ", ".join(rule.names)
        if rule.diameter_km is not None:
            read_from += (
                f", {rule.diameter_km[0]:g} to {rule.diameter_km[1]:g} km whole"
            )
        if rule.latitudes is not None:
            read_from += f", {rule.latitudes[0]:g} to {rule.latitudes[1]:g} deg"
        rows.append(
            (
                name,
                read_from,
                f"{held[name]:,}",
                f"{drawn.total():,}",
                ", ".join(f"{one} ({count})" for one, count in drawn.most_common()),
            )
        )
    return tables.written("Every class and what it is read from", _CLASSES, rows)


def crops(
    records: Sequence[ObservationMetadata], labels: Sequence[Label], pair: Sequence[str]
) -> widgets.Widget:
    """Tabulate what the crops of two classes hold, instrument by instrument."""
    classes = {one.tile: one.label for one in labels}
    rows: list[Row] = []
    for instrument in sorted({one.instrument for one in records}):
        for label in pair:
            held = [
                one
                for one in records
                if one.instrument == instrument and classes.get(one.tile) == label
            ]
            measured = [one for one in held if one.value_mean is not None]
            rows.append(
                (
                    instrument,
                    label,
                    f"{len(held):,}",
                    f"{np.mean([one.value_mean for one in measured]):.4g}"
                    if measured
                    else "",
                    f"{np.mean([one.value_std for one in measured]):.4g}"
                    if measured
                    else "",
                )
            )
    return tables.written("What the crops of the two classes hold", _CROPS, rows)
