"""Picking what is drawn, which is one whole tile and nothing else."""

from __future__ import annotations

from collections.abc import Callable

import ipywidgets as widgets
from IPython.display import display

from analysis import configs
from analysis.coverage.artifacts import index
from analysis.stats.artifacts import selection
from analysis.visualization.common import panels
from analysis.visualization.common.models.coverage import Coverage
from shared.maths import tessellate

DEFAULT_LAT = 18.4
DEFAULT_LON = 77.5
KEPT = "kept"
NO_WINDOW = "no window"
UNSEARCHED = "not in the selection"
COORDINATE = widgets.Layout(width="200px")


class TilePicker:
    """A picker taking a point on Mars, and the areas it fills below itself.

    Attributes:
        coverage: The confirmed tile's instrument sets, widest coverage first.
    """

    def __init__(self) -> None:
        """Build the picker, which reads nothing until a point is confirmed."""
        self.coverage: Coverage = []
        self._areas: list[tuple[widgets.Box, Callable[[Coverage], widgets.Widget]]] = []
        self._lat = widgets.BoundedFloatText(
            description="Latitude:",
            value=DEFAULT_LAT,
            min=-90.0,
            max=90.0,
            layout=COORDINATE,
        )
        self._lon = widgets.BoundedFloatText(
            description="Longitude:",
            value=DEFAULT_LON,
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
        self._areas = [claimed for claimed in self._areas if claimed[1] is not render]
        self._areas.append((area, render))
        display(area)
        area.children = (render(self.coverage),)

    def _confirmed(self, _button=None) -> None:
        """Load the tile holding the confirmed point and refill every claimed area."""
        settings = configs.load()
        tile = tessellate.tile_at(self._lat.value, self._lon.value, settings.tile_km)
        # The config says in what order the sets are drawn
        ranks = {
            chosen.key: rank for rank, chosen in enumerate(settings.instrument_sets)
        }
        self.coverage = sorted(
            index.load_tile(tile),
            key=lambda one: ranks.get(one.summary.set_key, len(ranks)),
        )
        try:
            picked = selection.selection_by_tile().get(tile.name)
        except FileNotFoundError:
            picked = None
        if picked is None:
            verdict = UNSEARCHED
        else:
            verdict = KEPT if picked.tile.kept else NO_WINDOW
        if self.coverage:
            note = widgets.HTML(
                f"Loaded <b>tile {tile.name}</b>, {tile.min_lat:.3f} to "
                f"{tile.max_lat:.3f} lat, {tile.west_lon:.3f} to {tile.east_lon:.3f} "
                f"lon, {verdict}. The cells below have filled in."
            )
        else:
            note = panels.unavailable(
                f"Nothing has been downloaded or computed for tile {tile.name}."
            )
        self._status.children = (note,)
        for area, _ in self._areas:
            area.children = ()
        for area, render in self._areas:
            area.children = (render(self.coverage),)
