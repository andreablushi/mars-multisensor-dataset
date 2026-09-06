"""What cleaning a CRISM multispectral survey observation is settled by."""

from __future__ import annotations

# What a wavelength file writes where the detector was never calibrated.
UNCALIBRATED = 65535.0

# What a label says about the ground calibration software rather than about the
# scan, which reads the same in every product it ever wrote.
GROUND_SOFTWARE = ("MRO:IKF_", "MRO:RSC_", "MRO:REFZ_", "MRO:FRAM_STAT_")

# Which detector's backplanes place a merged observation. Both halves are taken
# on the one grid, so one of them places every pixel of it.
PLACING_DETECTOR = "l"

# The wavelength window in nm each detector is trusted over, outside which the
# sensor edge sees almost no light and the reading is noise.
WINDOWS = {"l": (1020.0, 2650.0), "s": (400.0, 1060.0)}

# The range a brightness can take, its floor just below zero so noise there
# survives and only the impossible is refused.
BRIGHTNESS = (-0.05, 1.0)


# How wide in nm the moving median smoothing a column's spectrum reaches, fixed
# in nm so every downlink configuration means the same width.
STRIPE_WIDTH = 80.0

# How many deviations above its column's mean a band must sit to read as a
# spike, set per detector above the highest real absorption measured.
STRIPE_SIGMA = {"l": 7.5, "s": 4.7}

# Where the atmosphere absorbs inside each detector's window, in nm. Only the
# 2.0 um CO2 band costs less than it saves, the weaker ones sit on real features.
ATMOSPHERIC = {"l": ((1940.0, 2090.0),), "s": ()}

# The windows crism_ml despikes ratioed spectra with, in nm. The deviation is 20
# rather than 5, which keeps 99.7 per cent of a real one band absorption.
SPIKE_PASSES = ((72.0, 20.0), (46.0, 20.0), (20.0, 20.0))
