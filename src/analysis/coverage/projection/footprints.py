"""The tile box a footprint is cut to, and the ground each footprint covers."""

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

from analysis.coverage.models.region import TileRegion
from common.maths import geodesy, physics
from common.maths.geodesy import HALF_TURN, POLE, TURN, PolarGrid
from common.models.tile import Tile

POLAR_FOOTPRINT_DEG = 60.0

ODE_POLAR = {
    north: PolarGrid(0.0, north, physics.POLAR_RADIUS_M) for north in (True, False)
}

# Tracks are clipped to a dilated box so buffering still reaches the edge
CLIP_MARGIN_DEG = 2.0

# Straight lon/lat edges curve once projected, so resample below this step
MAX_SEGMENT_DEG = 0.25

MAX_SEGMENT_M = 1000.0

# Segments per quarter circle when a track is buffered to its swath
BUFFER_QUAD_SEGMENTS = 16

_EMPTY = Polygon()
_LINESTRING = 1
_POLYGON = 3
_FIRST_MULTIPART = 4


def tile_ring(tile: Tile) -> tuple[np.ndarray, np.ndarray]:
    """Return the tile's box as a closed lon/lat ring, densified to project smoothly."""
    return geodesy.bbox_ring(tile, MAX_SEGMENT_DEG)


def ring_polygon(x: np.ndarray, y: np.ndarray) -> BaseGeometry:
    """Close a projected ring into a polygon, repaired where it crosses itself.

    Args:
        x: The ring eastings in metres.
        y: The ring northings in metres.

    Returns:
        polygon: The valid polygon the ring bounds.
    """
    polygon = Polygon(np.column_stack((x, y)))
    # A box reaching every longitude crosses itself once projected
    if not is_valid(polygon):
        polygon = make_valid(polygon, method="structure", keep_collapsed=False)
    return polygon


def laea_tile(tile: Tile) -> BaseGeometry:
    """Project one tile's box onto the equal-area plane centred on the tile.

    Args:
        tile: The tile to project.

    Returns:
        shape: The box in equal-area metres, prepared for repeated queries.
    """
    shape = ring_polygon(
        *geodesy.laea_forward(*tile_ring(tile), tile.centre_lon, tile.centre_lat)
    )
    prepare(shape)
    return shape


def tile_region(tile: Tile) -> TileRegion:
    """Project one tile's box, with the lon/lat regions footprints are cut to.

    Args:
        tile: The tile whose box the coverage is measured against.

    Returns:
        region: The projected box and its clipping regions.
    """
    north = tile.min_lat >= 0.0
    polar_clip = polar_clip_wide = None
    if min(abs(tile.min_lat), abs(tile.max_lat)) >= POLAR_FOOTPRINT_DEG:
        polar_clip = ring_polygon(
            *geodesy.stereographic_forward(*tile_ring(tile), *ODE_POLAR[north])
        )
        polar_clip_wide = buffer(polar_clip, geodesy.northward_m(CLIP_MARGIN_DEG))
    return TileRegion(
        centre_lon=tile.centre_lon,
        centre_lat=tile.centre_lat,
        laea=laea_tile(tile),
        clip=clip_region(tile, 0.0),
        clip_wide=clip_region(tile, CLIP_MARGIN_DEG),
        polar_clip=polar_clip,
        polar_clip_wide=polar_clip_wide,
        north=north,
    )


def clip_region(tile: Tile, margin_deg: float) -> BaseGeometry:
    """Build the lon/lat region a footprint is cut against.

    Args:
        tile: The tile whose box is widened.
        margin_deg: How far to widen the box, in degrees of latitude.

    Returns:
        region: The clipping region, as one rectangle or the union of two.
    """
    lon_margin = margin_deg / geodesy.longitude_stretch(
        max(abs(tile.min_lat), abs(tile.max_lat))
    )
    lat_lo = max(-POLE, tile.min_lat - margin_deg)
    lat_hi = min(POLE, tile.max_lat + margin_deg)
    span = tile.span + 2.0 * lon_margin
    if span >= TURN:
        return box(-HALF_TURN, lat_lo, HALF_TURN, lat_hi)
    west = float(geodesy.normalise_longitude(tile.west_lon - lon_margin))
    east = west + span
    if east <= HALF_TURN:
        return box(west, lat_lo, east, lat_hi)
    return box(west, lat_lo, HALF_TURN, lat_hi).union(
        box(-HALF_TURN, lat_lo, east - TURN, lat_hi)
    )


def laea_footprints(
    region: TileRegion,
    geoms: np.ndarray,
    swath_widths_m: np.ndarray,
    *,
    stereographic: bool,
) -> np.ndarray:
    """Return the ground a whole set of observations covers on the tile.

    Args:
        region: The projected tile the footprints are cut to.
        geoms: The parsed footprints, in lon/lat or polar stereographic metres.
        swath_widths_m: The cross-track width for each track, ignored for areas.
        stereographic: Whether the geometries are in ODE's polar stereographic metres.

    Returns:
        footprints: One clipped footprint per input, empty where it falls outside.
    """
    if stereographic:
        clip_wide, clip, step = region.polar_clip_wide, region.polar_clip, MAX_SEGMENT_M
    else:
        clip_wide, clip, step = region.clip_wide, region.clip, MAX_SEGMENT_DEG
    parts, owners = single_parts(geoms)
    kinds = get_type_id(parts)
    # A footprint with any polygon is taken as areal, and its lines are dropped
    areal = np.zeros(len(geoms), dtype=bool)
    areal[owners[kinds == _POLYGON]] = True
    keep = np.where(areal[owners], kinds == _POLYGON, kinds == _LINESTRING)
    parts, owners = parts[keep], owners[keep]
    clips = np.asarray([clip_wide, clip], dtype=object)
    clipped = intersection(parts, clips[areal[owners].astype(int)])
    alive = ~is_empty(clipped)
    parts, owners = clipped[alive], owners[alive]
    radii = np.where(areal[owners], 0.0, np.asarray(swath_widths_m)[owners] / 2.0)

    projected = transform(
        segmentize(parts, step),
        lambda coords: np.column_stack(
            geodesy.laea_forward(
                *(
                    geodesy.stereographic_inverse(*coords.T, *ODE_POLAR[region.north])
                    if stereographic
                    else coords.T
                ),
                region.centre_lon,
                region.centre_lat,
            )
        ),
    )
    grown = radii > 0.0
    projected[grown] = buffer(
        projected[grown], radii[grown], quad_segs=BUFFER_QUAD_SEGMENTS
    )
    # A footprint reaching far around the projection centre crosses itself
    broken = ~is_valid(projected)
    projected[broken] = make_valid(
        projected[broken], method="structure", keep_collapsed=False
    )

    # Each footprint's parts are put back together as the one shape they were
    shapes = np.full(len(geoms), _EMPTY, dtype=object)
    order = np.argsort(owners, kind="stable")
    projected, owners = projected[order], owners[order]
    inputs = np.arange(shapes.size)
    starts = np.searchsorted(owners, inputs, side="left")
    ends = np.searchsorted(owners, inputs, side="right")
    counts = ends - starts
    shapes[counts == 1] = projected[starts[counts == 1]]
    for index in np.nonzero(counts > 1)[0]:
        shapes[index] = union_all(projected[starts[index] : ends[index]])
    # Whatever reached past the tile is cut back to it
    outside = ~covers(region.laea, shapes)
    shapes[outside] = intersection(shapes[outside], region.laea)
    return shapes


def single_parts(geoms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Expand geometries into their non-empty single-part pieces.

    Args:
        geoms: The geometries to expand, including nested collections.

    Returns:
        parts: The flat single-part geometries.
        owners: The index of the input each came from.
    """
    parts = np.asarray(geoms, dtype=object)
    owners = np.arange(parts.size)
    while (get_type_id(parts) >= _FIRST_MULTIPART).any():
        parts, index = get_parts(parts, return_index=True)
        owners = owners[index]
    alive = ~is_empty(parts)
    return parts[alive], owners[alive]
