"""Drawing the tiles each build covers out of what the selection kept."""

from __future__ import annotations

import random
from collections.abc import Collection, Sequence
from dataclasses import replace

from analysis.ground_truth.models.label import Label
from analysis.selector.models.selection import Selection
from building.models.settings import TrainingSettings
from common.maths import box


def draw_training(
    picked: Sequence[Selection],
    settings: TrainingSettings,
    labels: Sequence[Label],
    refused: Collection[str] = (),
) -> list[Selection]:
    """Keep the share of the kept tiles the training build covers, drawn at random.

    Args:
        picked: What the search left of every tile it searched.
        settings: The settled choices for the build, whose share sizes it.
        labels: Every labelled tile, whose drawn boxes training never touches.
        refused: The tiles the review refused, whose boxes training never touches.

    Returns:
        kept: The selections to build, in the order the selection was written.
    """
    kept = [one for one in picked if one.tile.kept]
    wanted = round(settings.share * len(kept))
    taken = range(len(kept))
    if wanted < len(kept):
        taken = sorted(random.Random(settings.seed).sample(taken, wanted))
    held_out = box.bounds_boxes(
        one for one in labels if one.drawn or one.tile in refused
    )
    # Held out after the draw, so a new evaluation draw never reshuffles training
    return [
        kept[at]
        for at in taken
        if not box.touching(box.bounds_box(kept[at].tile), held_out).any()
    ]


def draw_evaluation(
    picked: Sequence[Selection], labels: Sequence[Label]
) -> list[Selection]:
    """Keep the tiles the balanced draw took for the evaluation build.

    Args:
        picked: What the search left of every tile it searched.
        labels: Every labelled tile, the drawn ones marked so.

    Returns:
        kept: The selections to build, each cut to its label's box, in selection order.
    """
    drawn = {one.tile: one for one in labels if one.drawn}
    return [
        replace(one, tile=box.recut(one.tile, cut))
        for one in picked
        if (cut := drawn.get(one.tile.tile)) is not None
    ]
