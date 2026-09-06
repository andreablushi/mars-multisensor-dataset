"""Central configuration for the coverage analysis stage."""

from __future__ import annotations

SPEED_OF_LIGHT = 299_792_458.0

# The gravitational parameter for Mars, whose radius is named beside the planet
MARS_GM = 4.2828372e13

# SHARAD transmits 15-25 MHz; its centre sets the sounding wavelength
SHARAD_CENTRE_FREQUENCY_HZ = 20e6
SHARAD_WAVELENGTH_M = SPEED_OF_LIGHT / SHARAD_CENTRE_FREQUENCY_HZ

# Tracks are clipped to a dilated box so buffering still reaches the edge
LINE_CLIP_MARGIN_DEG = 2.0


# The union is kept per cell so each insert touches a small shape. A cell of
# this grid is not a unit of the survey: it exists only to keep the union small.
MIN_UNION_CELLS = 4
MAX_UNION_CELLS = 32

# How many observations a sector folds in before its union is rebuilt in one
UNION_CHUNK = 64

# A sector covered to within this share of what it could hold
SATURATION_TOLERANCE = 1e-12

# Grid an overlay is snapped to when exact arithmetic cannot node it.
SNAP_GRID_M = 1e-6

# A SHARAD sounding is as wide as its swath and as long as the spacing between
# traces in a focused radargram, which ODE does not publish.
SHARAD_ALONG_TRACK_M = 460.0

# Ground pixel size for the sets ODE publishes no map scale for, in metres.
# CRISM's survey modes are binned about ten times coarser than its targeted one.
# CTX publishes one for all but a handful of its records, and theirs is the
# median of the 497,279 that do.
FALLBACK_PIXEL_M = {"MRO/CRISM/TRDR:msp*": 180.0, "MRO/CTX/EDR": 5.4}

# How wide one block of the measurement grid is, in kilometres. A feature is
# given a block of cells for every block of ground this wide it spans, so a
# large feature is measured on a finer grid rather than a coarser one.
GRID_KM = 100

# Straight lon/lat edges curve once projected, so resample below this step
MAX_SEGMENT_DEG = 0.25

# Segments per quarter circle when a track is buffered to its swath.
BUFFER_QUAD_SEGMENTS = 16

# A footprint smaller than this share of one cell is given no cell at all,
# since a whole cell would credit it with ground it never reached
MIN_CELL_SHARE = 0.5
