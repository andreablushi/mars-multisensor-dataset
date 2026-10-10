"""The tables the evaluation notebooks write: the classes, and the review of them."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import ipywidgets as widgets

from analysis.ground_truth import catalogue
from analysis.ground_truth.draw import drawn_labels
from analysis.ground_truth.models.label import Label
from analysis.ground_truth.models.settings import GroundTruthSettings
from analysis.visualization import panels
from analysis.visualization.panels import Row

_CLASSES = ("Class", "Read from", "Tiles labelled", "Tiles drawn")
_VERDICTS = (
    "Class",
    "Tiles labelled",
    "Accepted",
    "Rejected",
    "Drawn now",
    "Drawn, not reviewed",
)


def classes(labels: Sequence[Label], settings: GroundTruthSettings) -> widgets.Widget:
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
        for low, high in rule.latitudes or []:
            read_from += f", {low:g} to {high:g} deg"
        rows.append((name, read_from, f"{labelled[name]:,}", f"{drawn[name]:,}"))
    return panels.written("Every class and what it is read from", _CLASSES, rows)


def verdicts(
    labels: Sequence[Label], settings: GroundTruthSettings, judged: dict[str, bool]
) -> widgets.Widget:
    """Tabulate every class's verdicts so far, and the draw they leave."""
    refused = catalogue.refused_tiles(judged)
    drawn = [label for label in drawn_labels(labels, settings, judged) if label.drawn]
    counts = [
        Counter(label.label for label in labels),
        Counter(label.label for label in labels if judged.get(label.tile)),
        Counter(label.label for label in labels if label.tile in refused),
        Counter(label.label for label in drawn),
        Counter(label.label for label in drawn if label.tile not in judged),
    ]
    rows = [
        (name, *(f"{count[name]:,}" for count in counts)) for name in settings.classes
    ]
    return panels.written("Every class, as the review leaves it", _VERDICTS, rows)
