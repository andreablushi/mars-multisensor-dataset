"""Drawing the tiles the evaluation build covers out of what the selection kept."""

from __future__ import annotations

from collections.abc import Sequence

from common.analysis.selector.models.selection import Selection
from evaluation.analysis.models.label import Label


def drawn_selections(
    picked: Sequence[Selection], labels: Sequence[Label]
) -> list[Selection]:
    """Keep the tiles the balanced draw took.

    Args:
        picked: What the search left of every tile it searched.
        labels: Every labelled tile, the drawn ones marked so.

    Returns:
        kept: The selections to build, in the order the selection was written.
    """
    drawn = {one.tile for one in labels if one.drawn}
    return [one for one in picked if one.tile.tile in drawn]
