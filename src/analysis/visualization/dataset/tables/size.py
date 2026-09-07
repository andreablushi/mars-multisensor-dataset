"""How big a dataset the filter leaves, and how much of it a build takes."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import ipywidgets as widgets

from analysis.selector.models.selection import Selection
from analysis.stats.models.dataset import DatasetStats
from analysis.visualization.common import quantities, tables, wording
from analysis.visualization.common.models.tables import Row

_HEADINGS = ("Statistic", "Value")
_AGAINST = ("Statistic", "The filter leaves", "A build takes")


def final(read: DatasetStats) -> widgets.Widget:
    """Tabulate the dataset the filter leaves behind."""
    held = read.held
    rows: list[Row] = [
        ("Features searched", f"{held.searched:,}"),
        ("Features kept", f"{held.kept:,}"),
        ("Mean window", wording.spread(held.days, quantities.duration)),
        ("Longest window", quantities.duration(held.days.high)),
    ]
    for iid in read.iids:
        measured = read.offered[iid]
        rows.append(
            (
                f"{iid} observations offered",
                f"{measured.mean * measured.counted:,.0f}",
            )
        )
    return tables.written("The dataset the filter leaves", _HEADINGS, rows)


def built(
    picked: Sequence[Selection], buildable: Sequence[Selection], cap: int
) -> widgets.Widget:
    """Tabulate what a build takes of the dataset once the crowded features are out."""
    kept = [one for one in picked if one.feature.kept]
    left, taken = (
        Counter(held.iid for one in group for held in one.observations)
        for group in (kept, buildable)
    )
    rows: list[Row] = [
        ("Features", f"{len(kept):,}", f"{len(buildable):,}"),
        (
            "Feature classes",
            f"{len({one.feature.feature_class for one in kept}):,}",
            f"{len({one.feature.feature_class for one in buildable}):,}",
        ),
        ("Observations", f"{sum(left.values()):,}", f"{sum(taken.values()):,}"),
    ]
    rows.extend(
        (f"{iid} observations", f"{left[iid]:,}", f"{taken[iid]:,}")
        for iid in sorted(left)
    )
    return tables.written(
        f"The dataset a build takes, at most {cap:,} observations a feature",
        _AGAINST,
        rows,
    )
