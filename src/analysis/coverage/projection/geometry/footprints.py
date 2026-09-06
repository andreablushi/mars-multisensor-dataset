"""The feature box a footprint is cut to, and the ground each footprint covers."""

from __future__ import annotations

import numpy as np
from shapely import (
    Polygon,
    box,
    buffer,
    covers,
    get_parts,
    get_type_id,
    intersection,
    is_empty,
    is_valid,
    make_valid,
    prepare,
    segmentize,
    transform,
    union_all,
)
from shapely.geometry.base import BaseGeometry

from analysis.coverage import configs
from analysis.coverage.models.region import FeatureRegion
from analysis.models.feature import Feature
from utils.geometry import geodesy

_EMPTY = Polygon()
_LINESTRING = 1
_POLYGON = 3
_FIRST_MULTIPART = 4


def feature_region(feature: Feature) -> FeatureRegion:
    """Project one feature's box, with the lon/lat regions footprints are cut to.

    Args:
        feature: The feature whose box the coverage is measured against.

    Returns:
        The projected box and the two clipping regions built from the same bounds.
    """
    min_lat, max_lat = feature.min_lat, feature.max_lat
    west_lon, east_lon = feature.west_lon, feature.east_lon
    centre_lon, centre_lat = geodesy.bbox_centre(min_lat, max_lat, west_lon, east_lon)
    lons, lats = geodesy.bbox_ring(
        min_lat, max_lat, west_lon, east_lon, configs.MAX_SEGMENT_DEG
    )
    x, y = geodesy.laea_forward(lons, lats, centre_lon, centre_lat)
    shape = Polygon(np.column_stack((x, y)))
    # A box reaching every longitude crosses itself once projected
    if not is_valid(shape):
        shape = make_valid(shape, method="structure", keep_collapsed=False)
    prepare(shape)
    return FeatureRegion(
        centre_lon=centre_lon,
        centre_lat=centre_lat,
        shape=shape,
        area_m2=shape.area,
        tight=clip_boxes(min_lat, max_lat, west_lon, east_lon),
        wide=clip_boxes(
            min_lat,
            max_lat,
            west_lon,
            east_lon,
            margin_deg=configs.LINE_CLIP_MARGIN_DEG,
        ),
    )


def clip_boxes(
    min_lat: float,
    max_lat: float,
    west_lon: float,
    east_lon: float,
    *,
    margin_deg: float = 0.0,
) -> BaseGeometry:
    """Build the lon/lat region a footprint is cut against.

    Args:
        min_lat: The southernmost latitude in degrees.
        max_lat: The northernmost latitude in degrees.
        west_lon: The westernmost longitude in degrees.
        east_lon: The easternmost longitude in degrees.
        margin_deg: How far to widen the region, in degrees of latitude.

    Returns:
        The clipping region, as one rectangle or the union of two.
    """
    lat_limit = min(max(abs(min_lat), abs(max_lat)), 89.0)
    lon_margin = margin_deg / geodesy.longitude_stretch(lat_limit)
    lat_lo = max(-90.0, min_lat - margin_deg)
    lat_hi = min(90.0, max_lat + margin_deg)
    span = geodesy.longitude_span(west_lon, east_lon) + 2.0 * lon_margin
    if span >= 360.0:
        return box(-180.0, lat_lo, 180.0, lat_hi)
    west = float(geodesy.normalise_longitude(west_lon - lon_margin))
    east = west + span
    if east <= 180.0:
        return box(west, lat_lo, east, lat_hi)
    return box(west, lat_lo, 180.0, lat_hi).union(
        box(-180.0, lat_lo, east - 360.0, lat_hi)
    )


def projected_footprints(
    region: FeatureRegion, geoms: np.ndarray, swath_widths_m: np.ndarray
) -> np.ndarray:
    """Return the ground a whole set of observations covers on the feature.

    Args:
        region: The projected feature the footprints are cut to.
        geoms: The parsed footprint geometries in lon/lat degrees.
        swath_widths_m: The cross-track width for each track, ignored for areas.

    Returns:
        One projected, clipped footprint per input, empty where it falls outside.
    """
    parts, owners = single_parts(geoms)
    kinds = get_type_id(parts)
    # A footprint with any polygon is taken as areal, and its lines are dropped
    areal = np.zeros(len(geoms), dtype=bool)
    areal[owners[kinds == _POLYGON]] = True
    keep = np.where(areal[owners], kinds == _POLYGON, kinds == _LINESTRING)
    parts, owners = parts[keep], owners[keep]
    regions = np.asarray([region.wide, region.tight], dtype=object)
    clipped = intersection(parts, regions[areal[owners].astype(int)])
    alive = ~is_empty(clipped)
    parts, owners = clipped[alive], owners[alive]
    radii = np.where(areal[owners], 0.0, np.asarray(swath_widths_m)[owners] / 2.0)

    projected = transform(
        segmentize(parts, configs.MAX_SEGMENT_DEG),
        lambda coords: np.column_stack(
            geodesy.laea_forward(
                coords[:, 0], coords[:, 1], region.centre_lon, region.centre_lat
            )
        ),
    )
    grown = radii > 0.0
    projected[grown] = buffer(
        projected[grown], radii[grown], quad_segs=configs.BUFFER_QUAD_SEGMENTS
    )
    # A footprint reaching far around the projection centre crosses itself
    broken = ~is_valid(projected)
    if broken.any():
        projected[broken] = make_valid(
            projected[broken], method="structure", keep_collapsed=False
        )

    # Each footprint's parts are put back together as the one shape they were
    shapes = np.full(len(geoms), _EMPTY, dtype=object)
    order = np.argsort(owners, kind="stable")
    projected, owners = projected[order], owners[order]
    wanted = np.arange(shapes.size)
    starts = np.searchsorted(owners, wanted, side="left")
    ends = np.searchsorted(owners, wanted, side="right")
    counts = ends - starts
    shapes[counts == 1] = projected[starts[counts == 1]]
    for index in np.nonzero(counts > 1)[0]:
        shapes[index] = union_all(projected[starts[index] : ends[index]])
    # Whatever reached past the feature is cut back to it
    outside = ~covers(region.shape, shapes)
    if outside.any():
        shapes[outside] = intersection(shapes[outside], region.shape)
    return shapes


def single_parts(geoms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Expand geometries into their non-empty single-part pieces.

    Args:
        geoms: The geometries to expand, including nested collections.

    Returns:
        The flat single-part geometries and the index of the input each came from.
    """
    parts = np.asarray(geoms, dtype=object)
    owners = np.arange(parts.size)
    while (get_type_id(parts) >= _FIRST_MULTIPART).any():
        parts, index = get_parts(parts, return_index=True)
        owners = owners[index]
    alive = ~is_empty(parts)
    return parts[alive], owners[alive]
