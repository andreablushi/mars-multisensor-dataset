"""Drawing the tiles the training build covers out of what the selection kept."""

from __future__ import annotations

import random
from collections.abc import Sequence, Set

from analysis.selector.models.selection import Selection
from building.models.settings import TrainingSettings


def drawn_selections(
    picked: Sequence[Selection], settings: TrainingSettings, held_out: Set[str]
) -> list[Selection]:
    """Keep the share of the kept tiles one build covers, drawn at random.

    Args:
        picked: What the search left of every tile it searched.
        settings: The settled choices for the build, whose share settles how much
            of what the filter kept one build covers.
        held_out: The tiles another dataset holds, which training never sees.

    Returns:
        kept: The selections to build, in the order the selection was written.
    """
    kept = [one for one in picked if one.tile.kept]
    wanted = round(settings.share * len(kept))
    taken = range(len(kept))
    if wanted < len(kept):
        taken = sorted(random.Random(settings.seed).sample(taken, wanted))
    # Held out after the draw, so a new evaluation draw never reshuffles training
    return [kept[at] for at in taken if kept[at].tile.tile not in held_out]
