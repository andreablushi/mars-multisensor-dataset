"""The filter every search runs under, the window section of the analysis config."""

from __future__ import annotations

from analysis import configs
from analysis.selector.models.filter import Filter

FILTER: Filter = configs.load().window
