"""How much of the ground each instrument has reached over time, and in total."""

from __future__ import annotations

import ipywidgets as widgets
from matplotlib.figure import Figure

from analysis.stats.instrument_sets import coverage_over_time
from analysis.visualization import panels
from analysis.visualization.panels import Coverage


def plot(coverage: Coverage) -> widgets.Widget:
    """Draw the running coverage of the whole tile, beside its total."""
    if not coverage:
        return panels.unavailable()
    timelines = coverage_over_time(coverage)
    colours = panels.colours([timeline.label for timeline in timelines])
    figure = Figure(figsize=(13, 5))
    running, bars = figure.subplots(1, 2, width_ratios=[3, 1])
    last = max(timeline.last for timeline in timelines)
    for timeline in timelines:
        if timeline.observed:
            times = [timeline.times[0], *timeline.times]
            fractions = [0.0, *timeline.running]
            if times[-1] < last:
                times.append(last)
                fractions.append(fractions[-1])
            style, note = "-", f"{timeline.covered:.1%}"
        else:
            times, fractions = [timeline.first, last], [0.0, 0.0]
            style, note = (0, (1, 3)), timeline.reason
        running.plot(
            times,
            fractions,
            linewidth=1.8,
            linestyle=style,
            color=colours[timeline.label],
            label=f"{timeline.label}  ({note})",
        )
    running.set_title(
        f"{panels.title(coverage)}  -  cumulative coverage", fontsize=12, loc="left"
    )
    running.set_xlabel("Observation start time")
    running.set_ylabel("Share of the tile covered so far")
    running.set_ylim(0, 1.05)
    running.set_xlim(right=last)
    panels.tidy(running, percent="y", grid="both")
    running.legend(fontsize=9, loc="upper left", frameon=False)
    ranked = timelines[::-1]
    bars.barh(
        [timeline.label for timeline in ranked],
        [timeline.covered for timeline in ranked],
        color=[colours[timeline.label] for timeline in ranked],
    )
    bars.set_title("Total covered", fontsize=11, loc="left")
    bars.set_xlim(0, 1.05)
    bars.tick_params(labelsize=9)
    panels.tidy(bars, percent="x", grid="x")
    figure.tight_layout()
    return panels.rendered(figure)
