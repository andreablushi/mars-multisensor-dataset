"""The review: every candidate crop of a region beside its cluttergram, to judge."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from functools import partial
from html import escape

import ipywidgets as widgets
import numpy as np
from matplotlib.figure import Figure

from mars_multisensor_dataset.analysis.ground_truth.lavaflow.labels import (
    write_lavaflow_verdicts,
)
from mars_multisensor_dataset.analysis.ground_truth.lavaflow.models import (
    LavaFlowLabel,
)
from mars_multisensor_dataset.analysis.utils.tile_group import tile_grid
from mars_multisensor_dataset.building import paths
from mars_multisensor_dataset.building.configs import sharad
from mars_multisensor_dataset.building.preprocessing.common import store
from mars_multisensor_dataset.building.preprocessing.sharad.window import (
    FEWEST_ROWS,
    rows_below_atmosphere,
)
from mars_multisensor_dataset.common.config import lavaflow_settings
from mars_multisensor_dataset.visualization import panels
from mars_multisensor_dataset.visualization.annotation_lavaflow import config
from mars_multisensor_dataset.visualization.labelled import TileStepper
from mars_multisensor_dataset.visualization.training_qualitative.preprocessing import (
    shown_crop,
    shown_range,
)


def plot(labels: Sequence[LavaFlowLabel], verdicts: dict[str, str]) -> widgets.Widget:
    """Step through every candidate crop of a region, track by track, to judge it."""
    ranked: dict[str, list[LavaFlowLabel]] = {}
    for label in labels:
        ranked.setdefault(label.region, []).append(label)
    if not ranked:
        return panels.unavailable(config.NOTHING_LISTED)
    return CropReview(ranked, verdicts).box


def table(labels: Sequence[LavaFlowLabel], verdicts: dict[str, str]) -> widgets.Widget:
    """Tabulate every region's verdicts so far."""
    sources = {label.region: label.source for label in labels}
    counts = [
        Counter(label.region for label in labels),
        Counter(label.region for label in labels if label.named),
        *(
            Counter(
                label.region for label in labels if verdicts.get(label.crop) == verdict
            )
            for verdict in config.BUTTON_STYLES
        ),
        Counter(label.region for label in labels if label.crop not in verdicts),
    ]
    rows = [
        (name, source, *(f"{count[name]:,}" for count in counts))
        for name, source in sources.items()
    ]
    return panels.written("Every region, as the review leaves it", config.REVIEW, rows)


def crop_panel(label: LavaFlowLabel) -> widgets.Widget:
    """Draw one crop's radargram beside its cluttergram, from its surface down."""
    frame = tile_grid().tile_named(label.tile)
    path = paths.dataset_root(lavaflow_settings().name) / store.sample_path(
        frame, sharad.LAYOUT.instrument, label.track
    )
    if not path.exists():
        return panels.unavailable(config.NOT_BUILT)
    with np.load(path) as crop:
        measured = crop[store.MEASURED]
        power = crop[sharad.LAYOUT.measurement].astype(np.float32)
        clutter = crop["clutter"].astype(np.float32)
    below = rows_below_atmosphere(power, measured)
    rows = slice(below.start, min(below.start + FEWEST_ROWS, below.stop))
    shown = (
        np.where(measured, power, np.nan),
        np.where(measured & (clutter > sharad.CLUTTER_FLOOR_DB), clutter, np.nan),
    )
    figure = Figure(figsize=config.FIGURE_SIZE, layout="constrained")
    figure.suptitle(
        f"{label.track} on {label.tile}, {frame.centre_lat:.2f}° N", fontsize=14
    )
    for axis, values, kind in zip(
        figure.subplots(1, 2), shown, config.KINDS, strict=True
    ):
        shown_crop(figure, axis, values, shown_range(values[rows]), kind, rows)
        axis.set_xlabel("Trace")
        axis.set_ylabel("Delay row")
    return panels.rendered(figure)


class CropReview(TileStepper):
    """The crops of every region, stepped through track by track and judged.

    Attributes:
        box: The review's controls, note and crop, as one widget.
    """

    def __init__(
        self, ranked: dict[str, list[LavaFlowLabel]], verdicts: dict[str, str]
    ) -> None:
        """Build the controls, the first region offered.

        Args:
            ranked: The crops of every region, each track south to north.
            verdicts: What each reviewed crop shows, written on every verdict.
        """
        self._verdicts = verdicts
        buttons = []
        for verdict, style in config.BUTTON_STYLES.items():
            button = widgets.Button(
                description=verdict.capitalize(), button_style=style
            )
            button.on_click(partial(self._judge, verdict))
            buttons.append(button)
        super().__init__(ranked, widgets.Box(), [widgets.HBox(buttons)])
        self._group.description = "Region:"
        self._tile.description = "Crop:"

    def _first(self, tiles: list[LavaFlowLabel]) -> int:
        """Return the index of the chosen region's first crop not yet reviewed."""
        return next(
            (at for at, label in enumerate(tiles) if label.crop not in self._verdicts),
            0,
        )

    def _name(self, label: LavaFlowLabel) -> str:
        """Return what the crop picker calls one crop."""
        return f"{label.track} on {label.tile}"

    def _drawn(self, label: LavaFlowLabel) -> widgets.Widget:
        """Draw one crop's radargram beside its cluttergram."""
        return crop_panel(label)

    def _place(self) -> str:
        """Return the chosen crop's place and verdict, and its region's count so far."""
        label = self._tile.value
        judged = Counter(
            self._verdicts.get(one.crop, config.NOT_REVIEWED)
            for one in self._ranked[label.region]
        )
        return (
            f"{self._tile.index + 1} of {len(self._tile.options)}, "
            f"{'named' if label.named else 'not named'} by {label.source}, "
            f"{self._verdicts.get(label.crop, config.NOT_REVIEWED)}. "
            f"{label.region}: "
            + ", ".join(
                f"{judged[held]:,} {held}"
                for held in (*config.BUTTON_STYLES, config.NOT_REVIEWED)
            )
        )

    def _judge(self, verdict: str, _button) -> None:
        """Write the verdict on the chosen crop, then move on to the next one.

        Args:
            verdict: What the crop shows.
        """
        self._verdicts[self._tile.value.crop] = verdict
        write_lavaflow_verdicts(self._verdicts)
        if self._tile.index + 1 < len(self._tile.options):
            self._tile.index += 1
        else:
            self._note.value = escape(self._place())
