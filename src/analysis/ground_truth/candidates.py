"""The candidate labels: each kept tile a single class claims, scored for the draw."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from analysis.ground_truth.claim import crater_claims, texture_claims
from analysis.ground_truth.classify import classified_features
from analysis.ground_truth.models.feature import ClassifiedFeatures, Feature
from analysis.ground_truth.models.label import Label
from analysis.ground_truth.models.settings import GroundTruthSettings
from analysis.selector.models.selection import SelectedTile
from common.maths import box


def candidate_labels(
    searched: Sequence[SelectedTile],
    features: Sequence[Feature],
    settings: GroundTruthSettings,
) -> list[Label]:
    """Label every kept tile a single class claims, and leave out every other.

    Args:
        searched: Every tile the selection searched.
        features: Every feature ODE publishes.
        settings: The settled choices for the labelling.

    Returns:
        labels: One label per tile a single class claims, in selection order.
    """
    kept = [tile for tile in searched if tile.kept]
    tile_boxes = box.bounds_boxes(kept)
    classified = classified_features(features, settings)
    craters = crater_claims(tile_boxes, classified, settings.classes)
    textures = texture_claims(tile_boxes, classified, settings.classes)
    labels = []
    for tile, kept_tile in enumerate(kept):
        claimed = craters[tile] if tile in craters else textures.get(tile, {})
        if len(claimed) != 1:
            continue
        ((label, (feature_index, offset)),) = claimed.items()
        feature = classified.features[feature_index]
        cut, own_class = (feature, None) if tile in craters else (kept_tile, label)
        labels.append(
            Label(
                tile=kept_tile.tile,
                label=label,
                feature=feature.name,
                overlaps=feature_overlaps(
                    classified, box.bounds_box(cut), feature.name, own_class
                ),
                centre_offset=offset,
                **box.box_edges(cut),
            )
        )
    return labels


def feature_overlaps(
    classified: ClassifiedFeatures, region: box.Box, name: str, own_class: str | None
) -> int:
    """Count the counted features touching a box, the labelled feature aside.

    Args:
        classified: Every feature, with its box, its classes and whether it counts.
        region: The box the label is cut to.
        name: The feature the label was read from.
        own_class: The texture class a feature may share and not count, or None for
            an object, which stands alone.

    Returns:
        overlaps: How many counted features touch the box against the label.
    """
    touched = np.flatnonzero(
        classified.counted & box.touching(classified.boxes, region)
    )
    # An object stands alone, while a texture may meet more of its own class
    return sum(
        classified.features[other].name != name
        and (own_class is None or classified.classes[other] != {own_class})
        for other in touched
    )
