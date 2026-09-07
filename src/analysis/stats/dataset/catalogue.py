"""Reading the measured dataset as one, before the filter is asked of it."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from analysis.coverage.artifacts import index
from analysis.coverage.models.summary import Summary
from analysis.metadata.loaders.features import load_features
from analysis.stats.models.catalogue import CatalogueStats, InstrumentStats
from analysis.stats.models.spread import Spread


def read_catalogue() -> CatalogueStats:
    """Read the catalogue index as one dataset.

    Returns:
        stats: What it holds, and nothing at all when no feature was measured.
    """
    catalogued = load_features()
    # One row per feature carries the grid, which every set of it shares
    by_feature: dict[tuple[str, str], Summary] = {}
    by_instrument: dict[str, list[Summary]] = {}
    for row in index.catalogued_rows():
        by_feature.setdefault((row.feature_class, row.feature_name), row)
        by_instrument.setdefault(row.iid, []).append(row)
    km2_by_class: dict[str, list[float]] = {}
    for (feature_class, _), row in by_feature.items():
        km2_by_class.setdefault(feature_class, []).append(row.feature_area_km2)
    return CatalogueStats(
        catalogued=len(catalogued),
        features=len(by_feature),
        points=sum(1 for feature in catalogued if feature.is_point),
        classes=dict(Counter(name for name, _ in by_feature).most_common()),
        class_km2={name: Spread.over(km2) for name, km2 in km2_by_class.items()},
        instruments=sorted(
            (_instrument(iid, rows) for iid, rows in by_instrument.items()),
            key=lambda instrument: -instrument.observations,
        ),
    )


def _instrument(iid: str, rows: Sequence[Summary]) -> InstrumentStats:
    """Read what one instrument holds of every feature it reached.

    Args:
        iid: The instrument the rows belong to.
        rows: Its rows, one per feature and instrument set it measured.

    Returns:
        stats: What it holds.
    """
    reached = {(row.feature_class, row.feature_name) for row in rows}
    return InstrumentStats(
        iid=iid,
        features=len(reached),
        observations=sum(row.n_obs for row in rows),
        first=min(row.t_first for row in rows),
        last=max(row.t_last for row in rows),
    )
