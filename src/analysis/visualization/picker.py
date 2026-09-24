"""Picking what is drawn, which is one whole tile and nothing else."""

from __future__ import annotations

from collections.abc import Callable

import ipywidgets as widgets
from IPython.display import display

from analysis.coverage import artifacts as index
from analysis.stats.artifacts import selection_by_tile
from analysis.utils.tile_group import tile_grid
from analysis.visualization import panels
from analysis.visualization.panels import Coverage
from common.config import analysis_settings

COORDINATE = widgets.Layout(width="200px")


class TilePicker:
    """A picker taking a point on Mars, and the areas it fills below itself.

    Attributes:
        coverage: The confirmed tile's instrument sets, widest coverage first.
    """

    def __init__(self) -> None:
        """Build the picker, which reads nothing until a point is confirmed."""
        self.coverage: Coverage = []
        self._areas: dict[Callable[[Coverage], widgets.Widget], widgets.Box] = {}
        self._lat = widgets.BoundedFloatText(
            description="Latitude:",
            value=18.4,
            min=-90.0,
            max=90.0,
            layout=COORDINATE,
        )
        self._lon = widgets.BoundedFloatText(
            description="Longitude:",
            value=77.5,
            min=-180.0,
            max=360.0,
            layout=COORDINATE,
        )
        self._confirm = widgets.Button(
            description="Confirm", button_style="primary", icon="check"
        )
        self._status = widgets.VBox()
        self._confirm.on_click(self._confirmed)

    def choose(self) -> None:
        """Display the picker."""
        controls = widgets.HBox([self._lat, self._lon, self._confirm])
        display(widgets.VBox([controls, self._status]))

    def show_panel(self, render: Callable[[Coverage], widgets.Widget]) -> None:
        """Claim an area here and fill it whenever the choice changes."""
        area = widgets.VBox()
        self._areas.pop(render, None)
        self._areas[render] = area
        display(area)
        area.children = (render(self.coverage),)

    def _confirmed(self, _button=None) -> None:
        """Load the tile holding the confirmed point and refill every claimed area."""
        grid = tile_grid()
        band, column = grid.tile_indices(self._lat.value, self._lon.value)
        tile = grid.tile_of(int(band), int(column))
        # The config says in what order the sets are drawn
        ranks = {
            chosen.key: rank
            for rank, chosen in enumerate(analysis_settings().instrument_sets)
        }
        self.coverage = sorted(
            index.load_tile(tile),
            key=lambda instrument: ranks.get(instrument.summary.set_key, len(ranks)),
        )
        try:
            selection = selection_by_tile().get(tile.name)
        except FileNotFoundError:
            selection = None
        if selection is None:
            verdict = "not in the selection"
        elif selection.tile.kept:
            verdict = "kept"
        else:
            verdict = "no window"
        if self.coverage:
            status = widgets.HTML(
                f"Loaded <b>tile {tile.name}</b>, {tile.min_lat:.3f} to "
                f"{tile.max_lat:.3f} lat, {tile.west_lon:.3f} to {tile.east_lon:.3f} "
                f"lon, {verdict}. The cells below have filled in."
            )
        else:
            status = panels.unavailable(
                f"Nothing has been downloaded or computed for tile {tile.name}."
            )
        self._status.children = (status,)
        for area in self._areas.values():
            area.children = ()
        for render, area in self._areas.items():
            area.children = (render(self.coverage),)
