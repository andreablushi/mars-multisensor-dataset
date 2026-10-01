"""The picker choosing the one tile every panel below it draws."""

from __future__ import annotations

from collections.abc import Callable

import ipywidgets as widgets
from IPython.display import display

from analysis.coverage import artifacts as coverage_artifacts
from analysis.stats.artifacts import read_tile_selection
from analysis.stats.order import config_rank
from analysis.utils.tile_group import tile_grid
from analysis.visualization import panels
from analysis.visualization.panels import Coverage
from common.maths import geodesy
from common.maths.box import POLE

COORDINATE = widgets.Layout(width="200px")


class TilePicker:
    """A picker taking a point on Mars, and the areas it fills below itself.

    Attributes:
        coverage: The confirmed tile's instrument sets, in the order the config
            draws them.
    """

    def __init__(self) -> None:
        """Build the picker, which reads nothing until a point is confirmed."""
        self.coverage: Coverage = []
        self._areas: dict[Callable[[Coverage], widgets.Widget], widgets.Box] = {}
        self._lat = widgets.BoundedFloatText(
            description="Latitude:",
            value=18.4,
            min=-POLE,
            max=POLE,
            layout=COORDINATE,
        )
        self._lon = widgets.BoundedFloatText(
            description="Longitude:",
            value=77.5,
            min=-geodesy.HALF_TURN,
            max=geodesy.TURN,
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
        """Claim an area here and fill it whenever the choice changes.

        Args:
            render: The panel the area is filled with.
        """
        area = widgets.VBox()
        self._areas.pop(render, None)
        self._areas[render] = area
        display(area)
        area.children = (self._panel(render),)

    def _panel(self, render: Callable[[Coverage], widgets.Widget]) -> widgets.Widget:
        """Draw one panel for the confirmed tile, or a stand-in where it has none."""
        return render(self.coverage) if self.coverage else panels.unavailable()

    def _confirmed(self, _button) -> None:
        """Load the tile holding the confirmed point and refill every claimed area."""
        grid = tile_grid()
        band, column = grid.tile_indices(self._lon.value, self._lat.value)
        tile = grid.tile_of(int(band), int(column))
        # The config says in what order the sets are drawn
        self.coverage = sorted(
            coverage_artifacts.read_tile_coverage(tile),
            key=lambda instrument: config_rank(instrument.summary.set_key),
        )
        if self.coverage:
            report = panels.tile_report(tile, read_tile_selection(tile.name))
            status = widgets.HTML(
                f"Loaded <b>tile {tile.name}</b>, {report}. "
                "The cells below have filled in."
            )
        else:
            status = panels.unavailable(
                f"Nothing has been downloaded or computed for tile {tile.name}."
            )
        self._status.children = (status,)
        for area in self._areas.values():
            area.children = ()
        for render, area in self._areas.items():
            area.children = (self._panel(render),)
