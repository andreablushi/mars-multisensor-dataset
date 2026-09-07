"""Every feature the selection searched, measured as the dataset stats read it."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor

from analysis.coverage.artifacts import index
from analysis.selector.models.selection import Selection
from analysis.stats.feature import measure
from analysis.stats.feature import read as feature
from analysis.stats.models.feature import FeatureStats

# Called with how many features are read and how many there are
Progress = Callable[[int, int], None]


def measure_every_feature(
    picked: Sequence[Selection], workers: int, progress: Progress | None = None
) -> list[FeatureStats]:
    """Measure what the selection kept of every feature it searched.

    Args:
        picked: What the search left of each feature, as the selection wrote it.
        workers: How many processes to measure on at once, as the run is configured.
        progress: Called with how many features are done and how many there are.

    Returns:
        measured: One entry per feature holding something to measure, in the order they
            came in, leaving out a feature with no measured set on disk.
    """
    found: list[FeatureStats] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for done, one in enumerate(
            pool.map(_measure_one_feature, picked, chunksize=1), 1
        ):
            if one is not None:
                found.append(one)
            if progress is not None:
                progress(done, len(picked))
    return found


def _measure_one_feature(picked: Selection) -> FeatureStats | None:
    """Measure one feature the selection searched.

    Args:
        picked: What the search left of it, and the observations it keeps.

    Returns:
        measured: What those looks leave on it, and None where it has no measured set on
            disk or holds nothing measurable.
    """
    coverage = index.load_feature(
        picked.feature.feature_class, picked.feature.feature_name
    )
    if not coverage:
        return None
    looks = feature.place_kept_looks(coverage, picked)
    return None if looks is None else measure.measured_feature(looks)
