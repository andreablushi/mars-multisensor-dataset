"""Labelling the tiles the selection kept by the features ODE places over them."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from analysis.labels import box
from analysis.labels.models.feature import Feature
from analysis.labels.models.label import Label
from analysis.labels.models.rule import Rule
from analysis.labels.models.settings import Settings
from analysis.selector.models.selection import SelectedTile
from common.maths.geodesy import northward_m


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
            and a texture tile is left out where another class claims it too or
            where any object of an object class reaches into it.
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
    claims: list[dict[str, str]] = [{} for _ in kept]
    objects = {
        label: rule
        for label, rule in settings.classes.items()
        if rule.diameter_km is not None
    }
    for label, rule in settings.classes.items():
        for feature in (one for one in features if read_from(rule, one)):
            if rule.diameter_km is not None:
                smallest, largest = rule.diameter_km
                diameter = northward_m(feature.max_lat - feature.min_lat) / 1000.0
                if not smallest <= diameter <= largest:
                    continue
                hit = box.inside(box.bounds_box(feature), tiles)
            elif (core := box.core_box(feature, settings.core, rule.latitudes)) is None:
                continue
            else:
                hit = box.inside(tiles, core)
            for at in np.flatnonzero(hit):
                claims[at].setdefault(label, feature.name)
    # Any object reaching into a texture tile, whatever its size, is ground of its own
    reached = np.zeros(len(kept), dtype=bool)
    for feature in features:
        if any(read_from(rule, feature) for rule in objects.values()):
            reached |= box.touching(tiles, box.bounds_box(feature))
    labels = []
    for at, held in enumerate(claims):
        chosen = {label: name for label, name in held.items() if label in objects}
        if not chosen and not reached[at]:
            chosen = held
        if len(chosen) == 1:
            ((label, name),) = chosen.items()
            labels.append(Label(tile=kept[at].tile, label=label, feature=name))
    return labels
