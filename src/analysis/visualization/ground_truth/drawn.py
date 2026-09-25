"""The drawn tiles of the evaluation set, one at a time, grouped by class."""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from html import escape

import ipywidgets as widgets

from analysis.ground_truth.models.label import Label
from analysis.ground_truth.models.settings import Settings
from analysis.visualization import mosaic, panels
from analysis.visualization.tile import basemap
from analysis.visualization.tile.placement import placed_tile

PICKER = widgets.Layout(width="360px")
STEP = widgets.Layout(width="40px")


def plot(labels: Sequence[Label], settings: Settings) -> widgets.Widget:
    """Step through the drawn tiles of every class over the mosaic."""
    grouped = {
        name: [label for label in labels if label.drawn and label.label == name]
        for name in settings.classes
    }
    grouped = {name: drawn for name, drawn in grouped.items() if drawn}
    if not grouped:
        return panels.unavailable("No tile has been drawn into the evaluation set.")
    group = widgets.Dropdown(options=list(grouped), description="Class:")
    tile = widgets.Dropdown(description="Tile:", layout=PICKER)
    previous = widgets.Button(icon="arrow-left", layout=STEP)
    following = widgets.Button(icon="arrow-right", layout=STEP)
    note = widgets.HTML()
    area = widgets.HBox(
        layout=widgets.Layout(align_items="flex-start", grid_gap="24px")
    )
    tile.observe(partial(show_tile, tile, note, area), names="value")
    group.observe(partial(offer_class, grouped, group, tile), names="value")
    previous.on_click(lambda _button: step_tile(tile, -1))
    following.on_click(lambda _button: step_tile(tile, 1))
    offer_class(grouped, group, tile)
    return widgets.VBox([widgets.HBox([group, tile, previous, following]), note, area])


def show_tile(
    tile: widgets.Dropdown, note: widgets.HTML, area: widgets.HBox, _change=None
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
    placed = placed_tile(label.tile, label)
    if placed is None:
        crop = panels.unavailable(mosaic.NO_BOX)
    else:
        title = f"Tile {label.tile}, {label.label}"
        crop = mosaic.fetched(
            placed.box(), lambda image: basemap.tile_map(placed, image, title)
        )
    area.children = (crop,)


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
