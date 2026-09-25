"""The frames footprints are projected in, and every constant shaping them."""

from __future__ import annotations

from common.maths import physics
from common.maths.geodesy import PolarGrid

POLAR_REACH_DEG = 60.0

# Tracks are clipped to a dilated box so buffering still reaches the edge
CLIP_MARGIN_DEG = 2.0

MAX_STRETCH_LAT_DEG = 89.0

# Straight lon/lat edges curve once projected, so resample below this step
MAX_SEGMENT_DEG = 0.25

MAX_SEGMENT_M = 1000.0

# Segments per quarter circle when a track is buffered to its swath
BUFFER_QUAD_SEGMENTS = 16


def ode_polar_grid(north: bool) -> PolarGrid:
    """Return ODE's polar stereographic frame over the north or the south pole."""
    return (0.0, north, physics.POLAR_RADIUS_M)
