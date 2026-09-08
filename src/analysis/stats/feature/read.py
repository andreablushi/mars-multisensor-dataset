"""One feature read back off the selection, on the timeline its looks sit on."""

from __future__ import annotations

from collections.abc import Sequence

from analysis.coverage.models.coverage import SetCoverage
from analysis.selector import configs as filtering
from analysis.selector.models import track as timeline
from analysis.selector.models.selection import Selection
from analysis.stats.artifacts import selection
from analysis.stats.models.feature import FeatureLooks

# How many features are held read at once, so every panel of one shares it.
FEATURE_CACHE = 8

# How many features are held read, so every panel of one shares the reading
_read: dict[tuple[str, str], FeatureLooks | None] = {}


def read_feature(coverage: Sequence[SetCoverage]) -> FeatureLooks | None:
    """Read one feature as the selection left it, however many panels ask for it.

    Args:
        coverage: The feature's instrument sets, in the order they are drawn.

    Returns:
        looks: Its timeline and the looks the selection keeps, or None where the
            selection never searched it or it holds nothing measurable.

    Raises:
        FileNotFoundError: When no selection has been written to read it off.
    """
    summary = coverage[0].summary
    key = (summary.feature_class, summary.feature_name)
    if key not in _read:
        if len(_read) >= FEATURE_CACHE:
            _read.clear()
        picked = selection.selection_by_feature().get(key)
        _read[key] = None if picked is None else place_kept_looks(coverage, picked)
    return _read[key]


def place_kept_looks(
    coverage: Sequence[SetCoverage], picked: Selection
) -> FeatureLooks | None:
    """Place the looks one feature keeps on the timeline they were taken over.

    Args:
        coverage: The feature's instrument sets, in any order.
        picked: What the selection left of it, and the observations it keeps.

    Returns:
        looks: Its timeline and where those looks sit on it, or None where the feature
            holds nothing measurable.
    """
    criteria, track = timeline.over(coverage, filtering.FILTER)
    if track is None:
        return None
    at = {one.pdsid: index for index, one in enumerate(track.observations)}
    return FeatureLooks(
        criteria=criteria,
        track=track,
        window=picked.feature,
        taken=tuple(
            sorted(at[one.pdsid] for one in picked.observations if one.pdsid in at)
        ),
    )
