"""The table the evaluation notebook writes: its classes, and what each holds."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import ipywidgets as widgets

from analysis.ground_truth.models.label import Label
from analysis.ground_truth.models.settings import Settings
from analysis.visualization import panels
from analysis.visualization.panels import Row

_CLASSES = ("Class", "Read from", "Tiles labelled", "Tiles drawn")


def classes(labels: Sequence[Label], settings: Settings) -> widgets.Widget:
    """Tabulate every class: what it is read from, and how many tiles it holds."""
    labelled = Counter(label.label for label in labels)
    drawn = Counter(label.label for label in labels if label.drawn)
    rows: list[Row] = []
    for name, rule in settings.classes.items():
        read_from = rule.descriptor or ", ".join(rule.names)
        if rule.diameter_km is not None:
            read_from += (
                f", {rule.diameter_km[0]:g} to {rule.diameter_km[1]:g} km whole"
            )
        if rule.latitudes is not None:
            read_from += f", {rule.latitudes[0]:g} to {rule.latitudes[1]:g} deg"
        rows.append((name, read_from, f"{labelled[name]:,}", f"{drawn[name]:,}"))
    return panels.written("Every class and what it is read from", _CLASSES, rows)
