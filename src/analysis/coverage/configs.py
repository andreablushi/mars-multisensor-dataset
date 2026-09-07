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


# The union is kept per cell so each insert touches a small shape, not as a unit
MIN_UNION_CELLS = 4
MAX_UNION_CELLS = 32

# How many observations a sector folds in before its union is rebuilt in one
UNION_CHUNK = 64

# A sector covered to within this share of what it could hold
SATURATION_TOLERANCE = 1e-12

# Grid an overlay is snapped to when exact arithmetic cannot node it.
SNAP_GRID_M = 1e-6

# A sounding is as wide as its swath and as long as a spacing ODE never publishes
SHARAD_ALONG_TRACK_M = 460.0

# Ground pixel size in metres for the sets ODE publishes no map scale for
FALLBACK_PIXEL_M = {"MRO/CRISM/TRDR:msp*": 180.0, "MRO/CTX/EDR": 5.4}

# How wide one block of the grid is, in kilometres, so large is not coarse
GRID_KM = 100

# Straight lon/lat edges curve once projected, so resample below this step
MAX_SEGMENT_DEG = 0.25

# Segments per quarter circle when a track is buffered to its swath.
BUFFER_QUAD_SEGMENTS = 16

# A footprint under this share of a cell is given none, to credit no ground
MIN_CELL_SHARE = 0.5
