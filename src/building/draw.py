"""Drawing the tiles each build covers out of what the selection kept."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from analysis.ground_truth import box
from analysis.ground_truth.models.label import Label
from analysis.selector.models.selection import Selection
from building.models.settings import TrainingSettings


def draw_training(
    picked: Sequence[Selection], settings: TrainingSettings, labels: Sequence[Label]
) -> list[Selection]:
    """Keep the share of the kept tiles the training build covers, drawn at random.

    Args:
        picked: What the search left of every tile it searched.
        settings: The settled choices for the build, whose share sizes it.
        labels: Every labelled tile, whose drawn boxes training never touches.

    Returns:
        kept: The selections to build, in the order the selection was written.
    """
    kept = [one for one in picked if one.tile.kept]
    wanted = round(settings.share * len(kept))
    taken = range(len(kept))
    if wanted < len(kept):
        taken = sorted(random.Random(settings.seed).sample(taken, wanted))
    held_out: box.Box = tuple(
        np.array(held)
        for held in zip(
            *(box.bounds_box(one) for one in labels if one.drawn), strict=True
        )
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
        replace(
            one,
            tile=replace(
                one.tile,
                min_lat=cut.min_lat,
                max_lat=cut.max_lat,
                west_lon=cut.west_lon,
                east_lon=cut.east_lon,
            ),
        )
        for one in picked
        if (cut := drawn.get(one.tile.tile)) is not None
    ]
