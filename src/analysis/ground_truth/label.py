"""Labelling the tiles the selection kept by the features ODE places over them."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from analysis.ground_truth import box
from analysis.ground_truth.models.feature import Feature
from analysis.ground_truth.models.label import Label
from analysis.ground_truth.models.rule import Rule
from analysis.ground_truth.models.settings import Settings
from analysis.selector.models.selection import SelectedTile
from common.maths.geodesy import TURN, northward_m
from common.maths.physics import METRES_PER_KM


def labelled_tiles(
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

    def read_from(rule: Rule, feature: Feature) -> bool:
        """Return whether one class is read from one feature.

        Args:
            rule: The class and what it is read from.
            feature: The feature to test.

        Returns:
            read: True when the feature is named by the class or carries its descriptor.
        """
        return feature.name in rule.names or feature.feature_class == rule.descriptor

    kept = [one for one in searched if one.kept]
    tiles: box.Box = tuple(
        np.array(held)
        for held in zip(*(box.bounds_box(one) for one in kept), strict=True)
    )
    owners = [
        {label for label, rule in settings.classes.items() if read_from(rule, one)}
        for one in features
    ]
    bounds: box.Box = tuple(
        np.array(held)
        for held in zip(*(box.bounds_box(one) for one in features), strict=True)
    )
    relevant = np.array(
        [
            bool(owned) or one.feature_class in settings.excluded
            for owned, one in zip(owners, features, strict=True)
        ]
    )
    claims: list[dict[str, tuple[int, float]]] = [{} for _ in kept]
    for at, feature in enumerate(features):
        for label in owners[at]:
            rule = settings.classes[label]
            if rule.diameter_km is not None:
                smallest, largest = rule.diameter_km
                diameter = (
                    northward_m(feature.max_lat - feature.min_lat) / METRES_PER_KM
                )
                if not smallest <= diameter <= largest:
                    continue
                south, north, west, span = box.bounds_box(feature)
                latitude, longitude = (south + north) / 2.0, (west + span / 2.0) % TURN
                hit = box.inside((latitude, latitude, longitude, 0.0), tiles)
                offset = np.zeros(len(kept))
            elif (claimed := box.claimed_box(feature, rule.latitudes)) is None:
                continue
            else:
                hit = box.inside(tiles, claimed)
                offset = box.centre_offset(tiles, claimed)
            for tile in np.flatnonzero(hit):
                claims[tile].setdefault(label, (at, float(offset[tile])))
    labels = []
    for tile, held in enumerate(claims):
        chosen = {
            label: claim
            for label, claim in held.items()
            if settings.classes[label].diameter_km is not None
        } or held
        if len(chosen) != 1:
            continue
        ((label, (at, offset)),) = chosen.items()
        name = features[at].name
        # An object stands alone, while a texture may meet more of its own class
        alone = settings.classes[label].diameter_km is not None
        cut = features[at] if alone else kept[tile]
        touched = np.flatnonzero(relevant & box.touching(bounds, box.bounds_box(cut)))
        foreign = sum(
            features[other].name != name and (alone or owners[other] != {label})
            for other in touched
        )
        labels.append(
            Label(
                tile=kept[tile].tile,
                label=label,
                feature=name,
                foreign=foreign,
                offset=offset,
                min_lat=cut.min_lat,
                max_lat=cut.max_lat,
                west_lon=cut.west_lon,
                east_lon=cut.east_lon,
            )
        )
    return labels
