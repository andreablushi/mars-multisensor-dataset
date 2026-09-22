"""Every feature of the catalogue read into the classes it belongs to."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from mars_multisensor_dataset.analysis.ground_truth.nomenclature.models.feature import (
    ClassifiedFeatures,
    Feature,
)
from mars_multisensor_dataset.common.maths import box

from .models.settings import (
    NomenclatureSettings,
)


def classified_features(
    features: Sequence[Feature], settings: NomenclatureSettings
) -> ClassifiedFeatures:
    """Read every feature into the classes it belongs to, and say which ones count.

    Args:
        features: Every feature ODE publishes.
        settings: The settled choices, naming every class and excluded descriptor.

    Returns:
        classified: The features, their boxes, their classes and whether each counts.
    """
    classes = [
        {
            label
            for label, rule in settings.classes.items()
            if feature.name in rule.names or feature.feature_class == rule.descriptor
        }
        for feature in features
    ]
    return ClassifiedFeatures(
        features=features,
        boxes=box.bounds_boxes(features),
        classes=classes,
        counted=np.array(
            [
                bool(feature_classes) or feature.feature_class in settings.excluded
                for feature_classes, feature in zip(classes, features, strict=True)
            ]
        ),
    )
