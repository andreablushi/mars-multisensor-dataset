"""Every instrument's observations of one feature, merged onto one time axis."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from analysis.coverage.models.coverage import Event, SetCoverage
from analysis.selector.filters import admissible, season
from analysis.selector.filters.clean_window import clean_window
from analysis.selector.models.filter import Filter
from analysis.selector.models.grid import Grid
from analysis.utils import mask as packing

# Seconds in a day, which is what every span is measured in.
DAY_SECONDS = 86400.0


@dataclass(frozen=True, slots=True)
class Track:
    """One feature's observations, from every instrument set, in time order.

    Attributes:
        observations: The observations the search may pick from, oldest first.
        times: When each of them started, in days, which is what a span is measured in.
        ls: How far round its year Mars had turned as each was taken, in degrees,
            which is what a window's width is held to.
        owners: The instrument set each belongs to, as its index into labels.
        cells: The feature's cells each fills, in the same order, each named once.
        labels: The name of each set, in the order owners index them.
        iids: The instrument each set belongs to, in the same order.
        grid: The grid the feature is searched over.
        refused: The observations left off the axis, each with the set it belongs
            to and the cells it fills, oldest first.
    """

    observations: list[Event]
    times: list[float]
    ls: list[float]
    owners: list[int]
    cells: list[np.ndarray]
    labels: list[str]
    iids: list[str]
    grid: Grid
    refused: admissible.Held


def build(
    coverage: Sequence[SetCoverage], grid: Grid, criteria: Filter
) -> Track | None:
    """Merge a feature's instrument sets onto one timeline.

    Args:
        coverage: The feature's instrument sets, in any order.
        grid: The grid the feature is searched over.
        criteria: The filter read against the feature, read once.

    Returns:
        track: The timeline, or None when the feature holds nothing measurable.
    """
    held, refused = admissible.admit_observation(coverage, grid, criteria)
    if not held:
        return None
    held.sort(key=lambda item: item[0].t_start)
    return Track(
        observations=[observation for observation, _, _ in held],
        times=[
            observation.t_start.timestamp() / DAY_SECONDS for observation, _, _ in held
        ],
        ls=season.arcs([observation.t_start for observation, _, _ in held]),
        owners=[owner for _, owner, _ in held],
        cells=[np.asarray(cells, dtype=np.intp) for _, _, cells in held],
        labels=[instrument.label for instrument in coverage],
        iids=[instrument.summary.iid for instrument in coverage],
        grid=grid,
        refused=sorted(refused, key=lambda item: item[0].t_start),
    )


def over(
    coverage: Sequence[SetCoverage], criteria: Filter
) -> tuple[Filter, Track | None]:
    """Read one feature's filter and timeline off the sets it holds.

    Args:
        coverage: The feature's instrument sets, in any order.
        criteria: Which instruments a window has to hold, and how much ground each.

    Returns:
        criteria: The filter as it was read against the feature.
        track: The timeline it is searched on, None where it holds nothing measurable.
    """
    summary = coverage[0].summary
    inside = packing.cells_of(summary.grid_mask).tolist()
    grid = Grid(
        cells=summary.grid_side * summary.grid_side,
        area_km2=len(inside) * summary.cell_km2,
        cell_km2=summary.cell_km2,
        inside=frozenset(inside),
    )
    # The one place the filter is read, which everything below takes it from
    settled = clean_window(criteria, coverage, grid)
    return settled, build(coverage, grid, settled)
