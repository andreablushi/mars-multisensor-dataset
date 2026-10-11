"""The review: every labelled tile of a class in draw order, and where it stands."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from functools import partial
from html import escape

import ipywidgets as widgets

from analysis.ground_truth import catalogue
from analysis.ground_truth.draw import drawn_labels, ranked_tiles
from analysis.ground_truth.models.label import Label
from analysis.ground_truth.models.settings import GroundTruthSettings
from visualization import panels
from visualization.annotation import config
from visualization.labelled import TileStepper


def plot(
    labels: Sequence[Label], settings: GroundTruthSettings, verdicts: dict[str, bool]
) -> widgets.Widget:
    """Step through every labelled tile of a class in draw order, to judge it."""
    by_tile = {label.tile: label for label in labels}
    ranked = {
        name: [by_tile[tile] for tile in tiles]
        for name, tiles in ranked_tiles(labels, settings, verdicts).items()
        if tiles
    }
    if not ranked:
        return panels.unavailable(config.NOTHING_LABELLED)
    return TileReview(ranked, verdicts).box


def table(
    labels: Sequence[Label], settings: GroundTruthSettings, verdicts: dict[str, bool]
) -> widgets.Widget:
    """Tabulate every class's verdicts so far, and the draw they leave."""
    refused = catalogue.refused_tiles(verdicts)
    drawn = [label for label in drawn_labels(labels, settings, verdicts) if label.drawn]
    counts = [
        Counter(label.label for label in labels),
        Counter(label.label for label in labels if verdicts.get(label.tile)),
        Counter(label.label for label in labels if label.tile in refused),
        Counter(label.label for label in drawn),
        Counter(label.label for label in drawn if label.tile not in verdicts),
    ]
    rows = [
        (name, *(f"{count[name]:,}" for count in counts)) for name in settings.classes
    ]
    return panels.written("Every class, as the review leaves it", config.VERDICTS, rows)


class TileReview(TileStepper):
    """The tiles of every class, stepped through in draw order and judged one by one.

    Attributes:
        box: The review's controls, note and tile, as one widget.
    """

    def __init__(
        self, ranked: dict[str, list[Label]], verdicts: dict[str, bool]
    ) -> None:
        """Build the controls, the first class offered.

        Args:
            ranked: The tiles of every class holding any, in draw order.
            verdicts: Whether each reviewed tile was accepted, written on every verdict.
        """
        self._verdicts = verdicts
        accept = widgets.Button(description="Accept", button_style="success")
        reject = widgets.Button(description="Reject", button_style="danger")
        accept.on_click(partial(self._judge, True))
        reject.on_click(partial(self._judge, False))
        super().__init__(ranked, widgets.Box(), [widgets.HBox([accept, reject])])

    def _first(self, tiles: list[Label]) -> int:
        """Return the index of the chosen class's first tile not yet reviewed."""
        return next(
            (at for at, label in enumerate(tiles) if label.tile not in self._verdicts),
            0,
        )

    def _place(self) -> str:
        """Return the chosen tile's place and verdict, and its class's count so far."""
        label = self._tile.value
        judged = Counter(
            self._verdicts.get(one.tile) for one in self._ranked[label.label]
        )
        return (
            f"{super()._place()}, {config.VERDICT[self._verdicts.get(label.tile)]}. "
            f"{label.label}: "
            + ", ".join(
                f"{judged[held]:,} {word}" for held, word in config.VERDICT.items()
            )
        )

    def _judge(self, accepted: bool, _button) -> None:
        """Write the verdict on the chosen tile, then move on to the next one.

        Args:
            accepted: Whether the tile is accepted.
        """
        self._verdicts[self._tile.value.tile] = accepted
        catalogue.write_verdicts(self._verdicts)
        if self._tile.index + 1 < len(self._tile.options):
            self._tile.index += 1
        else:
            self._note.value = escape(self._place())
