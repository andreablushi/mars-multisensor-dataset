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
from common.maths.geodesy import northward_m
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
        labels: One label per tile a single class claims, in the order the
            selection was written. A tile holding a whole object is that object,
            and a texture tile lies in the box of its feature, each counted for
            the features of other classes or excluded descriptors reaching into
            it, or of its own class where it holds an object.
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
    claims: list[dict[str, tuple[str, float]]] = [{} for _ in kept]
    touched: list[list[int]] = [[] for _ in kept]
    for at, feature in enumerate(features):
        if not owners[at] and feature.feature_class not in settings.excluded:
            continue
        bounds = box.bounds_box(feature)
        for tile in np.flatnonzero(box.touching(tiles, bounds)):
            touched[tile].append(at)
        for label in owners[at]:
            rule = settings.classes[label]
            if rule.diameter_km is not None:
                smallest, largest = rule.diameter_km
                diameter = (
                    northward_m(feature.max_lat - feature.min_lat) / METRES_PER_KM
                )
                if not smallest <= diameter <= largest:
                    continue
                hit = box.inside(bounds, tiles)
                offset = box.centre_offset(bounds, tiles)
            elif (claimed := box.claimed_box(feature, rule.latitudes)) is None:
                continue
            else:
                hit = box.inside(tiles, claimed)
                offset = box.centre_offset(tiles, claimed)
            for tile in np.flatnonzero(hit):
                claims[tile].setdefault(label, (feature.name, float(offset[tile])))
    labels = []
    for tile, held in enumerate(claims):
        chosen = {
            label: claim
            for label, claim in held.items()
            if settings.classes[label].diameter_km is not None
        } or held
        if len(chosen) != 1:
            continue
        ((label, (name, offset)),) = chosen.items()
        # An object stands alone, while a texture may meet more of its own class
        alone = settings.classes[label].diameter_km is not None
        foreign = sum(
            features[at].name != name and (alone or owners[at] != {label})
            for at in touched[tile]
        )
        labels.append(
            Label(
                tile=kept[tile].tile,
                label=label,
                feature=name,
                foreign=foreign,
                offset=offset,
            )
        )
    return labels
