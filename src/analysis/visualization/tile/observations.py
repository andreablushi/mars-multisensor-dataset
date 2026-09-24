"""What each single observation covered, at the time it was taken."""

from __future__ import annotations

import ipywidgets as widgets
from matplotlib.lines import Line2D

from analysis.stats.instrument_sets import coverage_over_time
from analysis.stats.tile import read_tile
from analysis.visualization import panels
from analysis.visualization.panels import Coverage
from analysis.visualization.tile.stems import stems

SURVEY_LINE = "#1a1a1a"
SURVEY_STYLE = (0, (6, 3))
SURVEY_WIDTH = 0.8


def plot(coverage: Coverage) -> widgets.Widget:
    """Draw one stacked panel per instrument set, over the whole tile."""
    if not coverage:
        return panels.unavailable()
    tile_looks = read_tile(coverage)
    open_for = tile_looks.open_for if tile_looks else []
    timeless = tile_looks.criteria.timeless if tile_looks else frozenset()
    timelines = coverage_over_time(coverage)
    colours = panels.colours([timeline.label for timeline in timelines])
    figure, axes = panels.stacked(
        len(timelines), 2.5 * len(timelines), sharex=True, sharey=True
    )
    for axis, timeline in zip(axes, timelines, strict=True):
        stems(axis, timeline.times, timeline.shares, colours[timeline.label])
        if not timeline.observed:
            panels.note(axis, timeline.reason)
        axis.set_ylabel(timeline.label, rotation=0, ha="right", va="center", fontsize=9)
        panels.tidy(axis, percent="y", grid="y")
        if timeline.iid in timeless:
            continue
        for opened, closed in open_for:
            axis.axvspan(opened, closed, color="#9e9e9e", alpha=0.18, zorder=0)
            for edge in (opened, closed):
                axis.axvline(
                    edge,
                    color=SURVEY_LINE,
                    linestyle=SURVEY_STYLE,
                    linewidth=SURVEY_WIDTH,
                    zorder=4,
                )
    if open_for and any(timeline.iid not in timeless for timeline in timelines):
        marker = Line2D(
            [],
            [],
            color=SURVEY_LINE,
            linestyle=SURVEY_STYLE,
            linewidth=SURVEY_WIDTH,
            label="the window the tile earned",
        )
        axes[0].legend(handles=[marker], fontsize=8, loc="upper right", frameon=False)
    if not any(timeline.observed for timeline in timelines):
        axes[0].set_xlim(
            min(timeline.first for timeline in timelines),
            max(timeline.last for timeline in timelines),
        )
    axes[0].set_ylim(-0.05, 1.05)
    title = f"{panels.title(coverage)}  -  coverage per observation"
    axes[0].set_title(title, fontsize=12, loc="left")
    axes[-1].set_xlabel("Observation start time")
    figure.supylabel("Share of the tile covered by one observation", fontsize=10)
    figure.tight_layout()
    return panels.rendered(figure)
