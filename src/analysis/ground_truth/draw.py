"""The balanced draw: the same number of every class, its features in turn."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace

from analysis.ground_truth.catalogue import refused_tiles
from analysis.ground_truth.models.label import Label
from analysis.ground_truth.models.settings import GroundTruthSettings


def ranked_tiles(
    labels: Sequence[Label], settings: GroundTruthSettings, verdicts: Mapping[str, bool]
) -> dict[str, list[str]]:
    """Rank the tiles of every class in the order the balanced draw takes them.

    Args:
        labels: Every labelled tile.
        settings: The settled choices, which name the classes and seed the ties.
        verdicts: Whether each reviewed tile was accepted, a refused one ranked last.

    Returns:
        ranked: The tiles of every class in config order, the first taken first.
    """
    refused = refused_tiles(verdicts)
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
            in_turn = sorted(
                feature_labels,
                key=lambda label: (
                    label.tile in refused,
                    not verdicts.get(label.tile, False),
                    label.overlaps,
                    label.centre_offset,
                ),
            )
            for turn, label in enumerate(in_turn):
                # One feature at a time in turn, so no single feature fills its class
                ranked.append(
                    (
                        label.tile in refused,
                        turn,
                        label.overlaps,
                        label.centre_offset,
                        rng.random(),
                        label.tile,
                    )
                )
        ranked_by_class[name] = [tile for *_, tile in sorted(ranked)]
    return ranked_by_class


def drawn_labels(
    labels: Sequence[Label], settings: GroundTruthSettings, verdicts: Mapping[str, bool]
) -> list[Label]:
    """Mark the tiles the balanced draw takes of every class, its features in turn.

    Args:
        labels: Every labelled tile.
        settings: The settled choices, which size the draw and seed it.
        verdicts: Whether each reviewed tile was accepted, a refused one never taken.

    Returns:
        labels: The same labels in the same order, the ones drawn marked so.

    Raises:
        ValueError: When a class holds fewer tiles than every class is drawn for.
    """
    refused = refused_tiles(verdicts)
    drawable = Counter(label.label for label in labels if label.tile not in refused)
    per_class = settings.per_class or min(drawable[name] for name in settings.classes)
    if short := {
        name: drawable[name] for name in settings.classes if drawable[name] < per_class
    }:
        raise ValueError(
            f"{per_class} tiles are drawn per class, but {short} hold fewer"
        )
    drawn: set[str] = set()
    for ranked in ranked_tiles(labels, settings, verdicts).values():
        drawn.update([tile for tile in ranked if tile not in refused][:per_class])
    return [replace(label, drawn=label.tile in drawn) for label in labels]
