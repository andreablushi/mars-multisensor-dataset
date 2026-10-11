"""The drawn tiles of the evaluation set, one at a time, grouped by class."""

from __future__ import annotations

from collections.abc import Sequence

import ipywidgets as widgets

from analysis.ground_truth.models.label import Label
from analysis.ground_truth.models.settings import GroundTruthSettings
from visualization import panels
from visualization.evaluation_quantitative import config
from visualization.labelled import TileStepper


def plot(labels: Sequence[Label], settings: GroundTruthSettings) -> widgets.Widget:
    """Step through the drawn tiles of every class over the mosaic."""
    grouped = drawn_by_class(labels, settings)
    if not grouped:
        return panels.unavailable(config.NOTHING_DRAWN)
    area = widgets.Layout(align_items="flex-start", grid_gap="24px")
    return TileStepper(grouped, widgets.HBox(layout=area)).box


def drawn_by_class(
    labels: Sequence[Label], settings: GroundTruthSettings
) -> dict[str, list[Label]]:
    """Group the drawn tiles of every class holding any, in config order."""
    return {
        name: drawn
        for name in settings.classes
        if (drawn := [label for label in labels if label.drawn and label.label == name])
    }
