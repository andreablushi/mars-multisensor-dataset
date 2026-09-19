"""Drawing the tiles the training build covers out of what the selection kept."""

from __future__ import annotations

import random
from collections.abc import Sequence

from common.analysis.selector.models.selection import Selection
from training.building.models.settings import Settings


def drawn_selections(
    picked: Sequence[Selection], settings: Settings
) -> list[Selection]:
    """Keep the share of the kept tiles one build covers, drawn at random.

    Args:
        picked: What the search left of every tile it searched.
        settings: The settled choices for the build, whose share settles how much
            of what the filter kept one build covers.

    Returns:
        kept: The selections to build, in the order the selection was written.
    """
    kept = [one for one in picked if one.tile.kept]
    wanted = round(settings.share * len(kept))
    if wanted >= len(kept):
        return kept
    taken = random.Random(settings.seed).sample(range(len(kept)), wanted)
    return [kept[at] for at in sorted(taken)]
