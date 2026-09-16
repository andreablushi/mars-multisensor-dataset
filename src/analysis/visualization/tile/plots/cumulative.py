"""How much of the ground each instrument has reached over time, and in total."""

from __future__ import annotations

import ipywidgets as widgets
from matplotlib.figure import Figure

from analysis.stats.tile import series
from analysis.visualization.common import panels
from analysis.visualization.common.models.coverage import Coverage

UNOBSERVED_LINESTYLE = (0, (1, 3))
CUMULATIVE_FIGURE_SIZE = (13, 5)
CUMULATIVE_WIDTH_RATIOS = [3, 1]

_GROUND = "Share of the tile covered so far"


def plot(coverage: Coverage) -> widgets.Widget:
    """Draw the running coverage of the whole tile, beside its total."""
    if not coverage:
        return panels.unavailable()
    drawn = series.coverage_over_time(coverage)
    colours = panels.colours([one.label for one in drawn])
    figure = Figure(figsize=CUMULATIVE_FIGURE_SIZE)
    running, bars = figure.subplots(1, 2, width_ratios=CUMULATIVE_WIDTH_RATIOS)
    last = max(one.last for one in drawn)
    for one in drawn:
        if one.observed:
            times = [one.times[0]] + list(one.times)
            fractions = [0.0] + list(one.running)
            if times[-1] < last:
                times.append(last)
                fractions.append(fractions[-1])
        else:
            times, fractions = [one.first, last], [0.0, 0.0]
        running.plot(
            times,
            fractions,
            linewidth=1.8,
            linestyle="-" if one.observed else UNOBSERVED_LINESTYLE,
            color=colours[one.label],
            label=(
                f"{one.label}  ({one.covered:.1%})"
                if one.observed
                else f"{one.label}  ({one.reason})"
            ),
        )
    running.set_title(
        f"{panels.title(coverage)}  -  cumulative coverage", fontsize=12, loc="left"
    )
    running.set_xlabel("Observation start time")
    running.set_ylabel(_GROUND)
    running.set_ylim(0, 1.05)
    running.set_xlim(right=last)
    panels.tidy(running, percent="y", grid="both")
    running.legend(fontsize=9, loc="upper left", frameon=False)
    ranked = list(drawn)[::-1]
    bars.barh(
        [one.label for one in ranked],
        [one.covered for one in ranked],
        color=[colours[one.label] for one in ranked],
    )
    bars.set_title("Total covered", fontsize=11, loc="left")
    bars.set_xlim(0, 1.05)
    bars.tick_params(labelsize=9)
    panels.tidy(bars, percent="x", grid="x")
    figure.tight_layout()
    return panels.rendered(figure)
