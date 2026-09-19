"""Drawing the same number of tiles out of every class, spread over its features."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import replace
from itertools import chain, islice, zip_longest

from evaluation.analysis.models.label import Label
from evaluation.analysis.models.settings import Settings


def drawn_labels(labels: Sequence[Label], settings: Settings) -> list[Label]:
    """Mark the tiles the balanced draw takes of every class.

    Args:
        labels: Every labelled tile.
        settings: The settled choices, which size the draw and seed it.

    Returns:
        labels: The same labels in the same order, the ones drawn marked so.

    Raises:
        ValueError: When a class holds fewer tiles than every class is drawn for.
    """
    classes: dict[str, dict[str, list[Label]]] = {
        rule.label: {} for rule in settings.rules
    }
    for one in labels:
        classes[one.label].setdefault(one.feature, []).append(one)
    held = {label: sum(map(len, by.values())) for label, by in classes.items()}
    wanted = settings.per_class or min(held.values())
    if short := {label: count for label, count in held.items() if count < wanted}:
        raise ValueError(f"{wanted} tiles are drawn per class, but {short} hold fewer")
    draw = random.Random(settings.seed)
    taken: set[str] = set()
    for by_feature in classes.values():
        shuffled = [draw.sample(one, len(one)) for one in by_feature.values()]
        draw.shuffle(shuffled)
        # One feature at a time in turn, so no single feature fills its class
        turns = chain.from_iterable(zip_longest(*shuffled))
        taken.update(one.tile for one in islice(filter(None, turns), wanted))
    return [replace(one, drawn=one.tile in taken) for one in labels]
