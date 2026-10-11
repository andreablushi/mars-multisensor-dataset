"""A labelled tile: the stepper through its class, and its mosaic crop."""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from html import escape

import ipywidgets as widgets

from analysis.ground_truth.models.label import Label
from visualization import config, mosaic, panels
from visualization.placement import placed_tile, tile_map


class TileStepper:
    """The tiles of every class, stepped through one by one over the mosaic.

    Attributes:
        box: The stepper's pickers, controls, note and tile, as one widget.
    """

    def __init__(
        self,
        ranked: dict[str, list[Label]],
        area: widgets.Box,
        controls: Sequence[widgets.Widget] = (),
    ) -> None:
        """Build the pickers, the first class offered.

        Args:
            ranked: The tiles of every class holding any, in the order stepped.
            area: Where the chosen tile's crop is drawn.
            controls: The widgets shown between the pickers and the note.
        """
        self._ranked = ranked
        self._area = area
        self._note = widgets.HTML()
        self._group = widgets.Dropdown(options=list(ranked), description="Class:")
        self._tile = widgets.Dropdown(description="Tile:", layout=config.PICKER)
        previous = widgets.Button(icon="arrow-left", layout=config.STEP)
        following = widgets.Button(icon="arrow-right", layout=config.STEP)
        previous.on_click(lambda _button: step_tile(self._tile, -1))
        following.on_click(lambda _button: step_tile(self._tile, 1))
        self._tile.observe(self._show, names="value")
        self._group.observe(self._offer, names="value")
        self._offer()
        pickers = widgets.HBox([self._group, self._tile, previous, following])
        self.box = widgets.VBox([pickers, *controls, self._note, area])

    def _first(self, tiles: list[Label]) -> int:
        """Return the index of the chosen class's tile shown first."""
        return 0

    def _offer(self, _change=None) -> None:
        """Offer the chosen class's tiles, opening on the first one to show."""
        tiles = self._ranked[self._group.value]
        self._tile.unobserve(self._show, names="value")
        self._tile.options = [(label.tile, label) for label in tiles]
        self._tile.index = self._first(tiles)
        self._tile.observe(self._show, names="value")
        self._show()

    def _show(self, _change=None) -> None:
        """Draw the chosen tile's mosaic crop and note what labelled it."""
        self._area.children = (label_map(self._tile.value),)
        self._note.value = escape(self._place())

    def _place(self) -> str:
        """Return the chosen tile's place in its class and its feature."""
        return (
            f"{self._tile.index + 1} of {len(self._tile.options)}, "
            f"{self._tile.value.feature}"
        )


def step_tile(tile: widgets.Dropdown, by: int) -> None:
    """Move to a neighbouring tile of the class, wrapping at its ends."""
    tile.index = (tile.index + by) % len(tile.options)


def label_map(label: Label) -> widgets.Widget:
    """Draw one labelled tile's mosaic crop, cut to its box and titled by its class."""
    placed = placed_tile(label.tile, label)
    if placed is None:
        return panels.unavailable(config.NO_BOX)
    title = f"Tile {label.tile}, {label.label}"
    return mosaic.fetched(placed.crop(), partial(tile_map, placed, title))
