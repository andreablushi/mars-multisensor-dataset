"""Laying one catalogued feature back onto the mosaic."""

from __future__ import annotations

from functools import lru_cache

from analysis.metadata.loaders.features import load_features
from analysis.models.feature import Feature
from analysis.visualization.feature.models.placing import Placed
from shared.disk.slugify import slugify

HALF_TURN_DEG = 180.0


def placed(feature_class: str, name: str) -> Placed | None:
    """Lay one feature back onto the mosaic.

    Args:
        feature_class: The feature class, such as Crater.
        name: The feature name as ODE spells it.

    Returns:
        placed: Where it falls in lon and lat, or None where no plate carree crop covers
            it.
    """
    grid = Placed(_catalogue()[slugify(feature_class), slugify(name)])
    # A feature wrapping the planet has no lon/lat box a plate carree crop can cover
    lon, _ = grid.outline()
    return grid if lon.max() - lon.min() <= HALF_TURN_DEG else None


@lru_cache(maxsize=1)
def _catalogue() -> dict[tuple[str, str], Feature]:
    """Read the feature catalogue once, keyed by the slugs the trees use.

    Returns:
        features: Every catalogued feature, by its class and name slug.
    """
    return {
        (slugify(feature.feature_class), slugify(feature.name)): feature
        for feature in load_features()
    }
