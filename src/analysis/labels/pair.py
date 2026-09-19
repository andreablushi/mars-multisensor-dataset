"""Drawing two classes of the evaluation set to be set side by side."""

from __future__ import annotations

import random
from collections.abc import Sequence

from analysis.labels.models.label import Label


def random_pair(labels: Sequence[Label], seed: int | None = None) -> tuple[str, str]:
    """Return two different classes of the set, drawn at random.

    Args:
        labels: The labels of the set.
        seed: The number the draw is made with, or None for a new one every time.

    Returns:
        first: One class.
        second: Another.
    """
    first, second = random.Random(seed).sample(sorted({one.label for one in labels}), 2)
    return first, second


def random_tiles(
    labels: Sequence[Label], pair: Sequence[str], seed: int | None = None
) -> list[str]:
    """Return one tile of each class of a pair, drawn at random.

    Args:
        labels: The labels of the set.
        pair: The classes to draw a tile of.
        seed: The number the draw is made with, or None for a new one every time.

    Returns:
        tiles: One tile per class, in the order the pair names them.
    """
    draw = random.Random(seed)
    return [
        draw.choice([one.tile for one in labels if one.label == label])
        for label in pair
    ]
