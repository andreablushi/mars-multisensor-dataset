"""The kept tiles each class claims: a crater by its centre, a texture by its region."""

from __future__ import annotations

import numpy as np

from analysis.ground_truth.models.class_rule import ClassRule
from analysis.ground_truth.models.feature import ClassifiedFeatures, Feature
from common.maths import box
from common.maths.geodesy import bbox_centre, northward_m
from common.maths.physics import METRES_PER_KM

Claims = dict[int, dict[str, tuple[int, float]]]


def crater_claims(
    tile_boxes: box.Box, classified: ClassifiedFeatures, classes: dict[str, ClassRule]
) -> Claims:
    """Claim every kept tile the centre of a crater sized for its class falls in.

    Args:
        tile_boxes: Every kept tile's box, stacked.
        classified: Every feature, with its box, its classes and whether it counts.
        classes: What every class is read from.

    Returns:
        claims: The first crater of every class claiming each tile, at offset 0.
    """
    crater_diameters = {
        label: rule.diameter_km
        for label, rule in classes.items()
        if rule.diameter_km is not None
    }
    claims: Claims = {}
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
                claims.setdefault(int(tile), {}).setdefault(label, (feature_index, 0.0))
    return claims


def texture_claims(
    tile_boxes: box.Box, classified: ClassifiedFeatures, classes: dict[str, ClassRule]
) -> Claims:
    """Claim every kept tile lying in a region of a texture class.

    Args:
        tile_boxes: Every kept tile's box, stacked.
        classified: Every feature, with its box, its classes and whether it counts.
        classes: What every class is read from.

    Returns:
        claims: The first feature of every class claiming each tile, and how far
            the tile sits from that region's centre.
    """
    texture_latitudes = {
        label: rule.latitudes
        for label, rule in classes.items()
        if rule.diameter_km is None
    }
    claims: Claims = {}
    for feature_index, feature in enumerate(classified.features):
        for label in classified.classes[feature_index] & texture_latitudes.keys():
            for region in texture_regions(feature, texture_latitudes[label]):
                offset = box.centre_offset(tile_boxes, region)
                for tile in np.flatnonzero(claimed_tiles(tile_boxes, region)):
                    claims.setdefault(int(tile), {}).setdefault(
                        label, (feature_index, float(offset[tile]))
                    )
    return claims


def texture_regions(
    feature: Feature, latitudes: list[list[float]] | None
) -> list[box.Box]:
    """Return the parts of a feature's box a texture tile has to lie in.

    Args:
        feature: The feature.
        latitudes: The latitude bands the class is kept to, or None for anywhere.

    Returns:
        regions: One box per band the feature reaches, its whole box for anywhere.
    """
    south, north, west, span = box.bounds_box(feature)
    return [
        (max(south, low), min(north, high), west, span)
        for low, high in latitudes or [[south, north]]
        if max(south, low) < min(north, high)
    ]


def claimed_tiles(tile_boxes: box.Box, region: box.Box) -> np.ndarray:
    """Return which tiles lie in a region, or centre in one too small to hold them.

    Args:
        tile_boxes: Every kept tile's box, stacked.
        region: The part of a feature's box a tile has to lie in.

    Returns:
        claimed: One flag per tile.
    """
    south, north, west, span = tile_boxes
    latitude = (south + north) / 2.0
    centre = (latitude, latitude, west + span / 2.0, 0.0)
    small = (region[1] - region[0] < 2 * (north - south)) | (region[3] < 2 * span)
    return box.inside(tile_boxes, region) | (small & box.inside(centre, region))
