"""The drawn tiles of the evaluation set, one at a time, grouped by class."""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from html import escape

import ipywidgets as widgets

from analysis.ground_truth.models.label import Label
from analysis.ground_truth.models.settings import GroundTruthSettings
from analysis.visualization import mosaic, panels
from analysis.visualization.tile import basemap
from analysis.visualization.tile.placement import placed_tile

PICKER = widgets.Layout(width="360px")
STEP = widgets.Layout(width="40px")
NOTHING_DRAWN = "No tile has been drawn into the evaluation set."


def plot(labels: Sequence[Label], settings: GroundTruthSettings) -> widgets.Widget:
    """Step through the drawn tiles of every class over the mosaic."""
    grouped = drawn_by_class(labels, settings)
    if not grouped:
        return panels.unavailable(NOTHING_DRAWN)
    group, tile, pickers = class_pickers(list(grouped))
    note = widgets.HTML()
    area = widgets.HBox(
        layout=widgets.Layout(align_items="flex-start", grid_gap="24px")
    )
    tile.observe(partial(show_tile, tile, note, area), names="value")
    group.observe(partial(offer_class, grouped, group, tile), names="value")
    offer_class(grouped, group, tile)
    return widgets.VBox([pickers, note, area])


def class_pickers(
    classes: Sequence[str],
) -> tuple[widgets.Dropdown, widgets.Dropdown, widgets.HBox]:
    """Build the class and tile pickers, beside the arrows stepping through tiles.

    Args:
        classes: The classes offered, in the order they are listed.

    Returns:
        group: The class picker.
        tile: The tile picker, offered nothing yet.
        row: Both pickers and the arrows, laid out in one row.
    """
    group = widgets.Dropdown(options=classes, description="Class:")
    tile = widgets.Dropdown(description="Tile:", layout=PICKER)
    previous = widgets.Button(icon="arrow-left", layout=STEP)
    following = widgets.Button(icon="arrow-right", layout=STEP)
    previous.on_click(lambda _button: step_tile(tile, -1))
    following.on_click(lambda _button: step_tile(tile, 1))
    return group, tile, widgets.HBox([group, tile, previous, following])


def drawn_by_class(
    labels: Sequence[Label], settings: GroundTruthSettings
) -> dict[str, list[Label]]:
    """Group the drawn tiles by their class.

    Args:
        labels: Every labelled tile, the drawn ones marked so.
        settings: The classes, in config order.

    Returns:
        grouped: The drawn tiles of every class holding any, in config order.
    """
    grouped = {
        name: [label for label in labels if label.drawn and label.label == name]
        for name in settings.classes
    }
    return {name: drawn for name, drawn in grouped.items() if drawn}


def show_tile(
    tile: widgets.Dropdown, note: widgets.HTML, area: widgets.HBox, _change
) -> None:
    """Draw the chosen tile's mosaic crop, noting what labelled it.

    Args:
        tile: The tile picker.
        note: Where the tile's place in its class and its feature are written.
        area: Where the crop is drawn.
    """
    label = tile.value
    if label is None:
        return
    note.value = escape(f"{tile.index + 1} of {len(tile.options)}, {label.feature}")
    area.children = (label_map(label),)


def label_map(label: Label) -> widgets.Widget:
    """Draw one labelled tile's mosaic crop, cut to its box and titled by its class."""
    placed = placed_tile(label.tile, label)
    if placed is None:
        return panels.unavailable(mosaic.NO_BOX)
    title = f"Tile {label.tile}, {label.label}"
    return mosaic.fetched(placed.crop(), partial(basemap.tile_map, placed, title))


def offer_class(
    grouped: dict[str, list[Label]],
    group: widgets.Dropdown,
    tile: widgets.Dropdown,
    _change=None,
) -> None:
    """Offer the tiles of the chosen class, the first of them shown.

    Args:
        grouped: The drawn tiles of every class holding any.
        group: The class picker.
        tile: The tile picker.
    """
    tile.options = [(label.tile, label) for label in grouped[group.value]]
    tile.index = 0


def step_tile(tile: widgets.Dropdown, by: int) -> None:
    """Move to a neighbouring tile of the class, wrapping at its ends.

    Args:
        tile: The tile picker.
        by: How many tiles to move, backwards when negative.
    """
    tile.index = (tile.index + by) % len(tile.options)
