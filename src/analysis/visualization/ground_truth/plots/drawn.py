"""The drawn tiles of the evaluation set, one at a time, grouped by class."""

from __future__ import annotations

from collections.abc import Sequence
from html import escape

import ipywidgets as widgets

from analysis.ground_truth.models.label import Label
from analysis.ground_truth.models.settings import Settings
from analysis.stats.artifacts import selection_by_tile
from analysis.visualization.common import mosaic, panels
from analysis.visualization.tile.plots import basemap, placing, radargram

PICKER = widgets.Layout(width="360px")
STEP = widgets.Layout(width="40px")


def plot(labels: Sequence[Label], settings: Settings) -> widgets.Widget:
    """Step through the drawn tiles of every class over the mosaic."""
    grouped = {
        name: [one for one in labels if one.drawn and one.label == name]
        for name in settings.classes
    }
    grouped = {name: held for name, held in grouped.items() if held}
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

    def show(_change=None) -> None:
        """Draw the chosen tile's radargram and mosaic crop, noting what labelled it."""
        one = tile.value
        if one is None:
            return
        note.value = escape(f"{tile.index + 1} of {len(tile.options)}, {one.feature}")
        grid = placing.placed(one.tile, one)
        if grid is None:
            crop = panels.unavailable(
                mosaic.BASEMAP_FAILED.format(reason=mosaic.NO_BOX)
            )
        else:
            box = grid.box()
            title = f"Tile {one.tile}, {one.label}"
            crop = mosaic.fetched(
                box, lambda image: basemap.figure(grid, box, image, title)
            )
        area.children = (radargram.plot(selection_by_tile()[one.tile]), crop)

    def regroup(_change=None) -> None:
        """Offer the tiles of the chosen class, the first of them shown."""
        tile.options = [(one.tile, one) for one in grouped[group.value]]
        tile.index = 0

    def step(by: int) -> None:
        """Move to a neighbouring tile of the class, wrapping at its ends."""
        tile.index = (tile.index + by) % len(tile.options)

    tile.observe(show, names="value")
    group.observe(regroup, names="value")
    previous.on_click(lambda _button: step(-1))
    following.on_click(lambda _button: step(1))
    regroup()
    return widgets.VBox([widgets.HBox([group, tile, previous, following]), note, area])
