"""The thresholds the best time window search itself turns on."""

from __future__ import annotations

# How far round its year Mars may turn for one more percentage point of ground,
# in degrees. Ten days at the mean rate, which is what the day priced window paid.
LS_PER_PERCENT = 5.25

# The cells an observation has to reach that no other observation of its own set
# already does, or it is dropped, as a share of the feature's own cells. A
# feature runs from a thousand cells to a hundred thousand of them, so a count
# asks a small landform for far more than a large one where a share asks both
# the same. At nought only a full repeat is dropped, and every higher value
# drops a look that does add ground.
GAIN_SHARE = 0.001

# Seconds in a day, which is what every span is measured in.
DAY_SECONDS = 86400.0
