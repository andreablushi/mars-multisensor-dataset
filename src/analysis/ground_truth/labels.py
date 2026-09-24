"""The kept tiles labelled by ODE's features, and the balanced set drawn from them."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Collection, Sequence
from dataclasses import replace
from operator import attrgetter

import numpy as np

from analysis.ground_truth.models.feature import Feature
from analysis.ground_truth.models.label import Label
from analysis.ground_truth.models.rule import Rule
from analysis.ground_truth.models.settings import Settings
from analysis.selector.models.selection import SelectedTile
from common.maths import box
from common.maths.geodesy import bbox_centre, northward_m
from common.maths.physics import METRES_PER_KM


def label_tiles(
    searched: Sequence[SelectedTile], features: Sequence[Feature], settings: Settings
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
    feature_classes = [
        {
            label
            for label, rule in settings.classes.items()
            if feature.name in rule.names or feature.feature_class == rule.descriptor
        }
        for feature in features
    ]
    feature_boxes = box.bounds_boxes(features)
    counted = np.array(
        [
            bool(classes) or feature.feature_class in settings.excluded
            for classes, feature in zip(feature_classes, features, strict=True)
        ]
    )
    crater_labels = label_craters(
        kept,
        tile_boxes,
        features,
        feature_boxes,
        feature_classes,
        counted,
        settings.classes,
    )
    texture_latitudes = {
        label: rule.latitudes
        for label, rule in settings.classes.items()
        if rule.diameter_km is None
    }
    claims: list[dict[str, tuple[int, float]]] = [{} for _ in kept]
    for feature_index, feature in enumerate(features):
        for label in feature_classes[feature_index] & texture_latitudes.keys():
            region = claimed_box(feature, texture_latitudes[label])
            if region is None:
                continue
            offset = box.centre_offset(tile_boxes, region)
            for tile in np.flatnonzero(box.inside(tile_boxes, region)):
                claims[tile].setdefault(label, (feature_index, float(offset[tile])))
    labels = []
    for tile, claimed in enumerate(claims):
        if tile in crater_labels:
            if crater_labels[tile] is not None:
                labels.append(crater_labels[tile])
            continue
        if len(claimed) != 1:
            continue
        ((label, (feature_index, offset)),) = claimed.items()
        name = features[feature_index].name
        kept_tile = kept[tile]
        touched = np.flatnonzero(
            counted & box.touching(feature_boxes, box.bounds_box(kept_tile))
        )
        labels.append(
            Label(
                tile=kept_tile.tile,
                label=label,
                feature=name,
                foreign=sum(
                    features[other].name != name and feature_classes[other] != {label}
                    for other in touched
                ),
                offset=offset,
                min_lat=kept_tile.min_lat,
                max_lat=kept_tile.max_lat,
                west_lon=kept_tile.west_lon,
                east_lon=kept_tile.east_lon,
            )
        )
    return labels


def label_craters(
    kept: Sequence[SelectedTile],
    tile_boxes: box.Box,
    features: Sequence[Feature],
    feature_boxes: box.Box,
    feature_classes: Sequence[set[str]],
    counted: np.ndarray,
    classes: dict[str, Rule],
) -> dict[int, Label | None]:
    """Label every kept tile the centre of a crater sized for its class falls in.

    Args:
        kept: Every tile the selection kept.
        tile_boxes: Their boxes, stacked.
        features: Every feature ODE publishes.
        feature_boxes: Their boxes, stacked.
        feature_classes: The classes each feature is read into.
        counted: Which features count against a label they touch.
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
    for feature_index, feature in enumerate(features):
        diameter = northward_m(feature.max_lat - feature.min_lat) / METRES_PER_KM
        sized_classes = [
            label
            for label in feature_classes[feature_index] & crater_diameters.keys()
            if crater_diameters[label][0] <= diameter <= crater_diameters[label][1]
        ]
        if not sized_classes:
            continue
        longitude, latitude = bbox_centre(
            feature.min_lat, feature.max_lat, feature.west_lon, feature.east_lon
        )
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
        crater = features[feature_index]
        # An object stands alone, while a texture may meet more of its own class
        touched = np.flatnonzero(
            counted & box.touching(feature_boxes, box.bounds_box(crater))
        )
        labels[tile] = Label(
            tile=kept[tile].tile,
            label=label,
            feature=crater.name,
            foreign=sum(features[other].name != crater.name for other in touched),
            offset=0.0,
            min_lat=crater.min_lat,
            max_lat=crater.max_lat,
            west_lon=crater.west_lon,
            east_lon=crater.east_lon,
        )
    return labels


def claimed_box(feature: Feature, latitudes: list[float] | None) -> box.Box | None:
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


def draw_labels(
    labels: Sequence[Label], settings: Settings, refused: Collection[str] = ()
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
    by_class: dict[str, dict[str, list[Label]]] = {
        name: {} for name in settings.classes
    }
    for label in labels:
        by_class[label.label].setdefault(label.feature, []).append(label)
    drawable = Counter(label.label for label in labels if label.tile not in refused)
    per_class = settings.per_class or min(drawable[name] for name in settings.classes)
    if short := {
        name: drawable[name] for name in settings.classes if drawable[name] < per_class
    }:
        raise ValueError(
            f"{per_class} tiles are drawn per class, but {short} hold fewer"
        )
    rng = random.Random(settings.seed)
    drawn: set[str] = set()
    for by_feature in by_class.values():
        ranked = []
        for feature_labels in by_feature.values():
            turns: Counter[int] = Counter()
            for label in sorted(feature_labels, key=attrgetter("foreign", "offset")):
                # One feature at a time in turn, so no single feature fills its class
                ranked.append(
                    (
                        label.foreign,
                        turns[label.foreign],
                        label.offset,
                        rng.random(),
                        label.tile,
                    )
                )
                turns[label.foreign] += 1
        # Skipped only once ranked, so a refusal never reshuffles what was accepted
        accepted = [tile for *_, tile in sorted(ranked) if tile not in refused]
        drawn.update(accepted[:per_class])
    return [replace(label, drawn=label.tile in drawn) for label in labels]
