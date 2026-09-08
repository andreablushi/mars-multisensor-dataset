"""Picking what is drawn, which is one whole feature and nothing else."""

from __future__ import annotations

from collections.abc import Callable

import ipywidgets as widgets
from IPython.display import display

from analysis import configs
from analysis.coverage.artifacts import index
from analysis.metadata.loaders.features import load_features
from analysis.stats.artifacts import selection
from analysis.visualization.common import panels
from analysis.visualization.common.models.coverage import Coverage
from shared.disk.slugify import slugify

DEFAULT_CLASS = "Crater"
NO_DATA_SUFFIX = "  (no data)"
KEPT_SUFFIX = "  (kept)"
NO_WINDOW_SUFFIX = "  (no window)"
DROPDOWN = widgets.Layout(width="300px")


class FeaturePicker:
    """A feature picker, and the areas it fills below itself.

    Attributes:
        coverage: The confirmed feature's instrument sets, widest coverage first.
    """

    def __init__(self) -> None:
        """Build the picker from the catalogue and what is computed on disk."""
        self._names: dict[str, list[str]] = {}
        for feature in load_features():
            self._names.setdefault(feature.feature_class, []).append(feature.name)
        for names in self._names.values():
            names.sort()
        self._computed = index.computed_features()
        self._kept = _kept_features()
        self.coverage: Coverage = []
        self._areas: list[tuple[widgets.Box, Callable[[Coverage], widgets.Widget]]] = []
        self._class = widgets.Dropdown(
            description="Type:",
            options=sorted(self._names),
            value=DEFAULT_CLASS,
            layout=DROPDOWN,
        )
        self._name = widgets.Dropdown(description="Name:", layout=DROPDOWN)
        self._confirm = widgets.Button(
            description="Confirm", button_style="primary", icon="check"
        )
        self._status = widgets.VBox()
        self._class.observe(self._refresh_names, names="value")
        self._confirm.on_click(self._confirmed)
        self._refresh_names()

    def choose(self) -> None:
        """Display the picker."""
        controls = widgets.HBox([self._class, self._name, self._confirm])
        display(widgets.VBox([controls, self._status]))

    def show_panel(self, render: Callable[[Coverage], widgets.Widget]) -> None:
        """Claim an area here and fill it whenever the choice changes."""
        area = widgets.VBox()
        self._areas = [claimed for claimed in self._areas if claimed[1] is not render]
        self._areas.append((area, render))
        display(area)
        area.children = (render(self.coverage),)

    def _refresh_names(self, _change=None) -> None:
        """Repopulate the name dropdown for the selected feature class."""
        feature_class = self._class.value

        def marked(name: str) -> str:
            """Say what a name is marked with: what it holds, then what it earned."""
            if (slugify(feature_class), slugify(name)) not in self._computed:
                return name + NO_DATA_SUFFIX
            kept = self._kept.get((feature_class, name))
            if kept is None:
                return name
            return name + (KEPT_SUFFIX if kept else NO_WINDOW_SUFFIX)

        self._name.options = [
            (marked(name), name) for name in self._names[feature_class]
        ]

    def _confirmed(self, _button=None) -> None:
        """Load the confirmed feature and refill every claimed area."""
        feature_class, name = self._class.value, self._name.value
        if (slugify(feature_class), slugify(name)) in self._computed:
            # The config says which sets are drawn, and in what order
            config = configs.load()
            wanted = config.plot_instrument_sets
            loaded = index.load_feature(feature_class, name)
            keys = {chosen.key for chosen in wanted or ()}
            kept = (
                list(loaded)
                if wanted is None
                else [one for one in loaded if one.summary.set_key in keys]
            )
            ranks = {
                chosen.key: rank
                for rank, chosen in enumerate(wanted or config.instrument_sets)
            }
            self.coverage = sorted(
                kept, key=lambda one: ranks.get(one.summary.set_key, len(ranks))
            )
            note = widgets.HTML(
                f"Loaded <b>{feature_class} / {name}</b>. "
                f"The cells below have filled in."
            )
        else:
            self.coverage = []
            note = panels.unavailable(
                f"Nothing has been downloaded or computed for {feature_class} / {name}."
            )
        self._status.children = (note,)
        for area, _ in self._areas:
            area.children = ()
        for area, render in self._areas:
            area.children = (render(self.coverage),)


def _kept_features() -> dict[tuple[str, str], bool]:
    """Say which searched features earned a window.

    Returns:
        kept: Whether each searched feature was kept, by class and name, and nothing at
            all where no selection has been written to read it off.
    """
    try:
        picked = selection.selection_by_feature()
    except FileNotFoundError:
        return {}
    return {key: one.feature.kept for key, one in picked.items()}
