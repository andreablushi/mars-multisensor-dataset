"""The thresholds the best time window search itself turns on."""

from __future__ import annotations

# How far Mars may turn for one more point of ground, in degrees; ten days at mean
LS_PER_PERCENT = 5.25

# The cells a look must bring that its own set has not, as a share of the feature
GAIN_SHARE = 0.001

# Seconds in a day, which is what every span is measured in.
DAY_SECONDS = 86400.0
