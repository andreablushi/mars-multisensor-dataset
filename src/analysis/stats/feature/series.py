"""One instrument set's observations of a feature, as a line over time."""

from __future__ import annotations

from collections.abc import Sequence

from analysis.coverage.models.coverage import SetCoverage
from analysis.stats.models.series import Series


def coverage_over_time(coverage: Sequence[SetCoverage]) -> list[Series]:
    """Read every instrument set's observations of the whole feature.

    Args:
        coverage: The feature's instrument sets, in the order they are drawn.

    Returns:
        series: One series per set, in the same order.
    """
    area_km2 = coverage[0].summary.feature_area_km2
    first = min(instrument.summary.t_first for instrument in coverage)
    last = max(instrument.summary.t_last for instrument in coverage)
    return [
        Series(
            label=instrument.label,
            iid=instrument.summary.iid,
            times=[observation.t_start for observation in instrument.events],
            shares=[
                observation.own_km2 / area_km2 for observation in instrument.events
            ],
            running=[observation.cum_frac for observation in instrument.events],
            covered=instrument.summary.covered_frac,
            first=first,
            last=last,
            reason=instrument.reason,
        )
        for instrument in coverage
    ]
