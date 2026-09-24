"""Labelling the kept tiles by ODE's features, and drawing the balanced set held out."""

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
    tiles = box.bounds_boxes(kept)
    owners = [
        {
            label
            for label, rule in settings.classes.items()
            if feature.name in rule.names or feature.feature_class == rule.descriptor
        }
        for feature in features
    ]
    bounds = box.bounds_boxes(features)
    relevant = np.array(
        [
            bool(owned) or feature.feature_class in settings.excluded
            for owned, feature in zip(owners, features, strict=True)
        ]
    )
    craters = label_craters(
        kept, tiles, features, bounds, owners, relevant, settings.classes
    )
    textures = {
        label: rule.latitudes
        for label, rule in settings.classes.items()
        if rule.diameter_km is None
    }
    claims: list[dict[str, tuple[int, float]]] = [{} for _ in kept]
    for at, feature in enumerate(features):
        for label in owners[at] & textures.keys():
            region = claimed_box(feature, textures[label])
            if region is None:
                continue
            offset = box.centre_offset(tiles, region)
            for tile in np.flatnonzero(box.inside(tiles, region)):
                claims[tile].setdefault(label, (at, float(offset[tile])))
    labels = []
    for tile, claimed in enumerate(claims):
        if tile in craters:
            if craters[tile] is not None:
                labels.append(craters[tile])
            continue
        if len(claimed) != 1:
            continue
        ((label, (at, offset)),) = claimed.items()
        name = features[at].name
        cut = kept[tile]
        touched = np.flatnonzero(relevant & box.touching(bounds, box.bounds_box(cut)))
        labels.append(
            Label(
                tile=cut.tile,
                label=label,
                feature=name,
                foreign=sum(
                    features[other].name != name and owners[other] != {label}
                    for other in touched
                ),
                offset=offset,
                min_lat=cut.min_lat,
                max_lat=cut.max_lat,
                west_lon=cut.west_lon,
                east_lon=cut.east_lon,
            )
        )
    return labels


def label_craters(
    kept: Sequence[SelectedTile],
    tiles: box.Box,
    features: Sequence[Feature],
    bounds: box.Box,
    owners: Sequence[set[str]],
    relevant: np.ndarray,
    classes: dict[str, Rule],
) -> dict[int, Label | None]:
    """Label every kept tile the centre of a crater sized for its class falls in.

    Args:
        kept: Every tile the selection kept.
        tiles: Their boxes, stacked.
        features: Every feature ODE publishes.
        bounds: Their boxes, stacked.
        owners: The classes each feature is read into.
        relevant: Which features count against a label they touch.
        classes: What every class is read from.

    Returns:
        craters: Each tile a crater claims, labelled, or None where two classes do.
    """
    sizes = {
        label: rule.diameter_km
        for label, rule in classes.items()
        if rule.diameter_km is not None
    }
    claims: dict[int, dict[str, int]] = {}
    for at, feature in enumerate(features):
        diameter = northward_m(feature.max_lat - feature.min_lat) / METRES_PER_KM
        sized = [
            label
            for label in owners[at] & sizes.keys()
            if sizes[label][0] <= diameter <= sizes[label][1]
        ]
        if not sized:
            continue
        longitude, latitude = bbox_centre(
            feature.min_lat, feature.max_lat, feature.west_lon, feature.east_lon
        )
        centre = (latitude, latitude, longitude, 0.0)
        for tile in np.flatnonzero(box.inside(centre, tiles)):
            for label in sized:
                claims.setdefault(int(tile), {}).setdefault(label, at)
    labels: dict[int, Label | None] = {}
    for tile, claimed in claims.items():
        if len(claimed) != 1:
            labels[tile] = None
            continue
        ((label, at),) = claimed.items()
        crater = features[at]
        # An object stands alone, while a texture may meet more of its own class
        touched = np.flatnonzero(
            relevant & box.touching(bounds, box.bounds_box(crater))
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
    wanted = settings.per_class or min(drawable[name] for name in settings.classes)
    if short := {
        name: drawable[name] for name in settings.classes if drawable[name] < wanted
    }:
        raise ValueError(f"{wanted} tiles are drawn per class, but {short} hold fewer")
    draw = random.Random(settings.seed)
    taken: set[str] = set()
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
                        draw.random(),
                        label.tile,
                    )
                )
                turns[label.foreign] += 1
        # Skipped only once ranked, so a refusal never reshuffles what was accepted
        kept = [tile for *_, tile in sorted(ranked) if tile not in refused]
        taken.update(kept[:wanted])
    return [replace(label, drawn=label.tile in taken) for label in labels]
