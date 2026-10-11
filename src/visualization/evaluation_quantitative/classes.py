"""Every class of the evaluation set: what it is read from, and where its tiles lie."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from functools import partial

import ipywidgets as widgets
from matplotlib.lines import Line2D

from analysis.ground_truth.models.label import Label
from analysis.ground_truth.models.settings import GroundTruthSettings
from common.maths import geodesy
from visualization import mosaic, panels
from visualization.config import MARS_PIXELS
from visualization.evaluation_quantitative import config
from visualization.evaluation_quantitative.drawn import drawn_by_class
from visualization.panels import Row


def table(labels: Sequence[Label], settings: GroundTruthSettings) -> widgets.Widget:
    """Tabulate every class: what it is read from, and its tiles and features."""
    labelled = [(label.label, label.feature) for label in labels]
    drawn = [(label.label, label.feature) for label in labels if label.drawn]
    counts = [
        Counter(name for name, _ in pairs)
        for pairs in (labelled, set(labelled), drawn, set(drawn))
    ]
    rows: list[Row] = []
    for name, rule in settings.classes.items():
        read_from = rule.descriptor or ", ".join(rule.names)
        if rule.diameter_km is not None:
            read_from += (
                f", {rule.diameter_km[0]:g} to {rule.diameter_km[1]:g} km whole"
            )
        for low, high in rule.latitudes or []:
            read_from += f", {low:g} to {high:g} deg"
        rows.append((name, read_from, *(f"{count[name]:,}" for count in counts)))
    return panels.written("Every class and what it is read from", config.CLASSES, rows)


def plot(labels: Sequence[Label], settings: GroundTruthSettings) -> widgets.Widget:
    """Map every drawn tile at its centre, one colour per class."""
    grouped = drawn_by_class(labels, settings)
    if not grouped:
        return panels.unavailable(config.NOTHING_DRAWN)
    return mosaic.fetched(mosaic.MARS, partial(classes_map, grouped), MARS_PIXELS)


def classes_map(grouped: dict[str, list[Label]], image: bytes) -> widgets.Image:
    """Draw the drawn tiles over the mosaic of Mars, one colour per class.

    Args:
        grouped: The tiles the balanced draw took, by class in config order.
        image: The mosaic of the whole planet, as fetched.

    Returns:
        map: The map, rendered.
    """
    drawn = sum(len(tiles) for tiles in grouped.values())
    figure, axis = mosaic.mars_board(image, f"{drawn:,} tiles drawn")
    colours = panels.colours(list(grouped))
    for name, tiles in grouped.items():
        lon, lat = zip(*(geodesy.bbox_centre(label) for label in tiles), strict=True)
        axis.scatter(
            lon,
            lat,
            s=config.MARKER_SIZE,
            color=colours[name],
            edgecolor="black",
            linewidth=0.4,
            transform=mosaic.LONLAT,
        )
    panels.key_beside(
        figure,
        [
            Line2D([], [], marker="o", linestyle="", color=colour, label=name)
            for name, colour in colours.items()
        ],
    )
    return panels.rendered(figure)
