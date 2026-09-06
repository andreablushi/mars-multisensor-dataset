"""What cleaning a CRISM multispectral survey observation is settled by."""

from __future__ import annotations

# What a wavelength file writes where the detector was never calibrated.
UNCALIBRATED = 65535.0

# What a label says about the calibration software, the same in every product.
GROUND_SOFTWARE = ("MRO:IKF_", "MRO:RSC_", "MRO:REFZ_", "MRO:FRAM_STAT_")

# Which detector places a merged observation, both halves being on the one grid.
PLACING_DETECTOR = "l"

# The nm window each detector is trusted over, outside which the reading is noise.
WINDOWS = {"l": (1020.0, 2650.0), "s": (400.0, 1060.0)}

# The range a brightness can take, its floor below zero so noise there survives.
BRIGHTNESS = (-0.05, 1.0)


# How wide the moving median reaches, in nm so every configuration means the same.
STRIPE_WIDTH = 80.0

# How far above its column's mean a band reads as a spike, set per detector.
STRIPE_SIGMA = {"l": 7.5, "s": 4.7}

# Where the atmosphere absorbs, in nm. Only the 2.0 um CO2 band is worth dropping.
ATMOSPHERIC = {"l": ((1940.0, 2090.0),), "s": ()}

# The windows crism_ml despikes with, in nm, at 20 deviations rather than five.
SPIKE_PASSES = ((72.0, 20.0), (46.0, 20.0), (20.0, 20.0))
