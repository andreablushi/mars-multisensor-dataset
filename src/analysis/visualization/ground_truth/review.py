"""The review: every labelled tile of a class in draw order, accepted or rejected."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from functools import partial
from html import escape

import ipywidgets as widgets

from analysis.ground_truth import catalogue
from analysis.ground_truth.draw import ranked_tiles
from analysis.ground_truth.models.label import Label
from analysis.ground_truth.models.settings import GroundTruthSettings
from analysis.visualization import panels
from analysis.visualization.ground_truth.drawn import class_pickers, label_map

VERDICT = {True: "accepted", False: "rejected", None: "not reviewed"}
NOTHING_LABELLED = "No tile has been labelled."


def plot(
    labels: Sequence[Label],
    settings: GroundTruthSettings,
    verdicts: dict[str, bool],
) -> widgets.Widget:
    """Step through every labelled tile of a class in draw order, to judge it."""
    by_tile = {label.tile: label for label in labels}
    ranked = {
        name: [by_tile[tile] for tile in tiles]
        for name, tiles in ranked_tiles(labels, settings, verdicts).items()
        if tiles
    }
    if not ranked:
        return panels.unavailable(NOTHING_LABELLED)
    return TileReview(ranked, verdicts).box


class TileReview:
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
        self._ranked = ranked
        self._verdicts = verdicts
        self._group, self._tile, pickers = class_pickers(list(ranked))
        accept = widgets.Button(description="Accept", button_style="success")
        reject = widgets.Button(description="Reject", button_style="danger")
        self._note = widgets.HTML()
        self._area = widgets.Box()
        self._tile.observe(self._show, names="value")
        self._group.observe(self._offer, names="value")
        accept.on_click(partial(self._judge, True))
        reject.on_click(partial(self._judge, False))
        self._offer()
        self.box = widgets.VBox(
            [
                pickers,
                widgets.HBox([accept, reject]),
                self._note,
                self._area,
            ]
        )

    def _offer(self, _change=None) -> None:
        """Offer the chosen class's tiles, opening on the first not yet reviewed."""
        tiles = self._ranked[self._group.value]
        self._tile.unobserve(self._show, names="value")
        self._tile.options = [(label.tile, label) for label in tiles]
        self._tile.index = next(
            (at for at, label in enumerate(tiles) if label.tile not in self._verdicts),
            0,
        )
        self._tile.observe(self._show, names="value")
        self._show()

    def _show(self, _change=None) -> None:
        """Draw the chosen tile's mosaic crop and note where the review stands."""
        if self._tile.value is None:
            return
        self._area.children = (label_map(self._tile.value),)
        self._write_note()

    def _write_note(self) -> None:
        """Note the chosen tile's feature and verdict, and its class's count so far."""
        label = self._tile.value
        judged = Counter(
            self._verdicts.get(one.tile) for one in self._ranked[label.label]
        )
        self._note.value = escape(
            f"{self._tile.index + 1} of {len(self._tile.options)}, {label.feature}, "
            f"{VERDICT[self._verdicts.get(label.tile)]}. {label.label}: "
            + ", ".join(f"{judged[held]:,} {word}" for held, word in VERDICT.items())
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
            self._write_note()
