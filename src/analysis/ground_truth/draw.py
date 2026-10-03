"""The balanced draw: the same number of every class, the clearest tiles first."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Collection, Sequence
from dataclasses import replace
from operator import attrgetter

from analysis.ground_truth.models.label import Label
from analysis.ground_truth.models.settings import GroundTruthSettings


def ranked_tiles(
    labels: Sequence[Label], settings: GroundTruthSettings
) -> dict[str, list[str]]:
    """Rank the tiles of every class in the order the balanced draw takes them.

    Args:
        labels: Every labelled tile.
        settings: The settled choices, which name the classes and seed the ties.

    Returns:
        ranked: The tiles of every class in config order, the first taken first.
    """
    by_class: dict[str, dict[str, list[Label]]] = {
        name: {} for name in settings.classes
    }
    for label in labels:
        by_class[label.label].setdefault(label.feature, []).append(label)
    rng = random.Random(settings.seed)
    ranked_by_class: dict[str, list[str]] = {}
    for name, by_feature in by_class.items():
        ranked = []
        for feature_labels in by_feature.values():
            turns: Counter[int] = Counter()
            for label in sorted(
                feature_labels, key=attrgetter("overlaps", "centre_offset")
            ):
                # One feature at a time in turn, so no single feature fills its class
                ranked.append(
                    (
                        label.overlaps,
                        turns[label.overlaps],
                        label.centre_offset,
                        rng.random(),
                        label.tile,
                    )
                )
                turns[label.overlaps] += 1
        ranked_by_class[name] = [tile for *_, tile in sorted(ranked)]
    return ranked_by_class


def drawn_labels(
    labels: Sequence[Label], settings: GroundTruthSettings, refused: Collection[str]
) -> list[Label]:
    """Mark the tiles the balanced draw takes of every class, the clearest first.

    Args:
        labels: Every labelled tile.
        settings: The settled choices, which size the draw and seed it.
        refused: The tiles the review refused, never taken.

    Returns:
        labels: The same labels in the same order, the ones drawn marked so.

    Raises:
        ValueError: When a class holds fewer tiles than every class is drawn for.
    """
    drawable = Counter(label.label for label in labels if label.tile not in refused)
    per_class = settings.per_class or min(drawable[name] for name in settings.classes)
    if short := {
        name: drawable[name] for name in settings.classes if drawable[name] < per_class
    }:
        raise ValueError(
            f"{per_class} tiles are drawn per class, but {short} hold fewer"
        )
    drawn: set[str] = set()
    for ranked in ranked_tiles(labels, settings).values():
        # Skipped only once ranked, so a refusal never reshuffles what was accepted
        accepted = [tile for tile in ranked if tile not in refused]
        drawn.update(accepted[:per_class])
    return [replace(label, drawn=label.tile in drawn) for label in labels]
