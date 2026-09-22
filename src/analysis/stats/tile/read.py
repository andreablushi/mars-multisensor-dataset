"""One tile read back off the selection, on the timeline its looks sit on."""

from __future__ import annotations

from collections.abc import Sequence

from analysis.coverage.models.coverage import SetCoverage
from analysis.selector import configs as filtering
from analysis.selector.models import track as timeline
from analysis.selector.models.selection import Selection
from analysis.stats.artifacts import selection
from analysis.stats.models.tile import TileLooks

# How many tiles are held read at once, so every panel of one shares it.
TILE_CACHE = 8

# How many tiles are held read, so every panel of one shares the reading
_read: dict[str, TileLooks | None] = {}


def read_tile(coverage: Sequence[SetCoverage]) -> TileLooks | None:
    """Read one tile as the selection left it, however many panels ask for it.

    Args:
        coverage: The tile's instrument sets, in the order they are drawn.

    Returns:
        looks: Its timeline and kept looks, or None where nothing is measurable.

    Raises:
        FileNotFoundError: When no selection has been written to read it off.
    """
    key = coverage[0].summary.tile
    if key not in _read:
        if len(_read) >= TILE_CACHE:
            _read.clear()
        picked = selection.selection_by_tile().get(key)
        _read[key] = None if picked is None else place_kept_looks(coverage, picked)
    return _read[key]


def place_kept_looks(
    coverage: Sequence[SetCoverage], picked: Selection
) -> TileLooks | None:
    """Place the looks one tile keeps on the timeline they were taken over.

    Args:
        coverage: The tile's instrument sets, in any order.
        picked: What the selection left of it, and the observations it keeps.

    Returns:
        looks: Its timeline and where its looks sit, or None if nothing measurable.
    """
    criteria, track = timeline.over(coverage, filtering.FILTER)
    if track is None:
        return None
    at = {one.pdsid: index for index, one in enumerate(track.observations)}
    return TileLooks(
        criteria=criteria,
        track=track,
        window=picked.tile,
        taken=tuple(
            sorted(at[one.pdsid] for one in picked.observations if one.pdsid in at)
        ),
    )
