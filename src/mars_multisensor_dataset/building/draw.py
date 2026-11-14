"""Drawing the tiles each build covers out of what the selection kept."""

from __future__ import annotations

import random
from collections.abc import Collection, Sequence
from dataclasses import replace

from mars_multisensor_dataset.analysis.ground_truth.lavaflow.models import (
    LavaFlowLabel,
)
from mars_multisensor_dataset.analysis.ground_truth.nomenclature.models.label import (
    Label,
)
from mars_multisensor_dataset.analysis.selector.models.selection import (
    SelectedObservation,
    Selection,
)
from mars_multisensor_dataset.building.configs import sharad
from mars_multisensor_dataset.building.models.settings import TrainingSettings
from mars_multisensor_dataset.common.maths import box


def draw_training(
    picked: Sequence[Selection],
    settings: TrainingSettings,
    labels: Sequence[Label],
    refused: Collection[str],
) -> list[Selection]:
    """Keep the share of the kept tiles the training build covers, drawn at random.

    Args:
        picked: What the search left of every tile it searched.
        settings: The settled choices for the build, whose share sizes it.
        labels: Every labelled tile, whose drawn boxes training never touches.
        refused: The tiles the review refused, whose own and label boxes training
            never touches.

    Returns:
        kept: The selections to build, in the order the selection was written.
    """
    kept = [one for one in picked if one.tile.kept]
    wanted = round(settings.share * len(kept))
    taken = range(len(kept))
    if wanted < len(kept):
        taken = sorted(random.Random(settings.seed).sample(taken, wanted))
    held_out = box.bounds_boxes(
        [one for one in labels if one.drawn or one.tile in refused]
        + [one.tile for one in picked if one.tile.tile in refused]
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
    """Keep the tiles the draw took for the evaluation build.

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


def draw_lavaflow(
    picked: Sequence[Selection], labels: Sequence[LavaFlowLabel]
) -> list[Selection]:
    """Keep every tile a lava flow crop lies on, holding its labelled tracks alone.

    Args:
        picked: What the search left of every tile it searched, kept or not.
        labels: Every candidate crop, one per track and tile.

    Returns:
        kept: The selections to build, each tile's own observations replaced by
            the tracks labelled on it, in selection order.
    """
    tracks: dict[str, list[SelectedObservation]] = {}
    for one in labels:
        tracks.setdefault(one.tile, []).append(
            SelectedObservation(
                tile=one.tile,
                **sharad.ODE,
                pt=sharad.PRODUCT_TYPES[sharad.Kind.OBSERVATION],
                pdsid=one.pdsid,
                t_start=one.t_start,
                standing=True,
            )
        )
    return [
        replace(one, observations=tracks[one.tile.tile])
        for one in picked
        if one.tile.tile in tracks
    ]
