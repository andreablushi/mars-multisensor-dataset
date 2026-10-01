"""How many features touch a candidate tile, counted against it in the draw."""

from __future__ import annotations

from collections.abc import Collection, Sequence

import numpy as np

from analysis.ground_truth.models.feature import ClassifiedFeatures, Feature
from common.maths import box


def counted_features(
    features: Sequence[Feature], classes: Sequence[set[str]], excluded: Collection[str]
) -> np.ndarray:
    """Return which features count against a label they touch: a class or descriptor."""
    return np.array(
        [
            bool(feature_classes) or feature.feature_class in excluded
            for feature_classes, feature in zip(classes, features, strict=True)
        ]
    )


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
