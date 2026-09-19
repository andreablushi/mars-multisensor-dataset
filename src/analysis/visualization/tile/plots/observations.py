"""What each single observation covered, at the time it was taken."""

from __future__ import annotations

import ipywidgets as widgets
from matplotlib.lines import Line2D

from analysis.stats.tile import read, series
from analysis.visualization.common import panels
from analysis.visualization.common.models.coverage import Coverage

PANEL_HEIGHT = 2.5

_GROUND = "Share of the tile covered by one observation"


def plot(coverage: Coverage) -> widgets.Widget:
    """Draw one stacked panel per instrument set, over the whole tile."""
    if not coverage:
        return panels.unavailable()
    looks = read.read_tile(coverage)
    open_for = looks.open_for if looks else []
    timeless = looks.criteria.timeless if looks else frozenset()
    drawn = series.coverage_over_time(coverage)
    colours = panels.colours([one.label for one in drawn])
    figure, axes = panels.stacked(
        len(drawn), PANEL_HEIGHT * len(drawn), sharex=True, sharey=True
    )
    marked = False
    for axis, one in zip(axes, drawn, strict=True):
        colour = colours[one.label]
        axis.vlines(one.times, 0.0, one.shares, color=colour, alpha=0.35, linewidth=0.7)
        axis.scatter(
            one.times,
            one.shares,
            s=12,
            alpha=0.65,
            color=colour,
            edgecolors="none",
            zorder=3,
        )
        if not one.observed:
            panels.note(axis, one.reason)
        axis.set_ylabel(one.label, rotation=0, ha="right", va="center", fontsize=9)
        panels.tidy(axis, percent="y", grid="y")
        if one.iid not in timeless:
            panels.shade(axis, open_for)
            marked = True
    if marked and open_for:
        marker = Line2D(
            [],
            [],
            color=panels.SURVEY_LINE,
            linestyle=panels.SURVEY_STYLE,
            linewidth=panels.SURVEY_WIDTH,
            label="the window the tile earned",
        )
        axes[0].legend(handles=[marker], fontsize=8, loc="upper right", frameon=False)
    if not any(one.observed for one in drawn):
        axes[0].set_xlim(
            min(one.first for one in drawn), max(one.last for one in drawn)
        )
    axes[0].set_ylim(-0.05, 1.05)
    title = f"{panels.title(coverage)}  -  coverage per observation"
    axes[0].set_title(title, fontsize=12, loc="left")
    axes[-1].set_xlabel("Observation start time")
    figure.supylabel(_GROUND, fontsize=10)
    figure.tight_layout()
    return panels.rendered(figure)
