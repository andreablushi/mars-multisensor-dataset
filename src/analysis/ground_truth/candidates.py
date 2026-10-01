"""The kept tiles each class claims: a crater by its centre, a texture by its region."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from analysis.ground_truth import overlaps
from analysis.ground_truth.models.class_rule import ClassRule
from analysis.ground_truth.models.feature import ClassifiedFeatures, Feature
from analysis.ground_truth.models.label import Label
from analysis.ground_truth.models.settings import GroundTruthSettings
from analysis.selector.models.selection import SelectedTile
from common.maths import box
from common.maths.geodesy import bbox_centre, northward_m
from common.maths.physics import METRES_PER_KM


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
    craters = crater_candidates(kept, tile_boxes, classified, settings.classes)
    texture_latitudes = {
        label: rule.latitudes
        for label, rule in settings.classes.items()
        if rule.diameter_km is None
    }
    claims: list[dict[str, tuple[int, float]]] = [{} for _ in kept]
    for feature_index, feature in enumerate(classified.features):
        for label in classified.classes[feature_index] & texture_latitudes.keys():
            region = texture_region(feature, texture_latitudes[label])
            if region is None:
                continue
            offset = box.centre_offset(tile_boxes, region)
            for tile in np.flatnonzero(box.inside(tile_boxes, region)):
                claims[tile].setdefault(label, (feature_index, float(offset[tile])))
    labels = []
    for tile, claimed in enumerate(claims):
        if tile in craters:
            if craters[tile] is not None:
                labels.append(craters[tile])
            continue
        if len(claimed) != 1:
            continue
        ((label, (feature_index, offset)),) = claimed.items()
        kept_tile = kept[tile]
        name = classified.features[feature_index].name
        labels.append(
            Label(
                tile=kept_tile.tile,
                label=label,
                feature=name,
                overlaps=overlaps.feature_overlaps(
                    classified, box.bounds_box(kept_tile), name, label
                ),
                centre_offset=offset,
                **box.box_edges(kept_tile),
            )
        )
    return labels


def classified_features(
    features: Sequence[Feature], settings: GroundTruthSettings
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
        counted=overlaps.counted_features(features, classes, settings.excluded),
    )


def crater_candidates(
    kept: Sequence[SelectedTile],
    tile_boxes: box.Box,
    classified: ClassifiedFeatures,
    classes: dict[str, ClassRule],
) -> dict[int, Label | None]:
    """Label every kept tile the centre of a crater sized for its class falls in.

    Args:
        kept: Every tile the selection kept.
        tile_boxes: Their boxes, stacked.
        classified: Every feature, with its box, its classes and whether it counts.
        classes: What every class is read from.

    Returns:
        labels: Each tile a crater claims, labelled, or None where two classes do.
    """
    crater_diameters = {
        label: rule.diameter_km
        for label, rule in classes.items()
        if rule.diameter_km is not None
    }
    claims: dict[int, dict[str, int]] = {}
    for feature_index, feature in enumerate(classified.features):
        diameter = northward_m(feature.max_lat - feature.min_lat) / METRES_PER_KM
        sized_classes = [
            label
            for label in classified.classes[feature_index] & crater_diameters.keys()
            if crater_diameters[label][0] <= diameter <= crater_diameters[label][1]
        ]
        if not sized_classes:
            continue
        longitude, latitude = bbox_centre(feature)
        centre = (latitude, latitude, longitude, 0.0)
        for tile in np.flatnonzero(box.inside(centre, tile_boxes)):
            for label in sized_classes:
                claims.setdefault(int(tile), {}).setdefault(label, feature_index)
    labels: dict[int, Label | None] = {}
    for tile, claimed in claims.items():
        if len(claimed) != 1:
            labels[tile] = None
            continue
        ((label, feature_index),) = claimed.items()
        crater = classified.features[feature_index]
        labels[tile] = Label(
            tile=kept[tile].tile,
            label=label,
            feature=crater.name,
            overlaps=overlaps.feature_overlaps(
                classified, box.bounds_box(crater), crater.name, None
            ),
            centre_offset=0.0,
            **box.box_edges(crater),
        )
    return labels


def texture_region(feature: Feature, latitudes: list[float] | None) -> box.Box | None:
    """Return the part of a feature's box a texture tile has to lie in.

    Args:
        feature: The feature.
        latitudes: The latitudes the class is kept to, or None for anywhere.

    Returns:
        box: The box, or None where the latitudes leave none of it.
    """
    south, north, west, span = box.bounds_box(feature)
    if latitudes is not None:
        south, north = max(south, latitudes[0]), min(north, latitudes[1])
    return (south, north, west, span) if south < north else None
