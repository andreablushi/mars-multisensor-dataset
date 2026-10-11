"""What each instrument reached of the tile on show: per look, over time, in pixels."""

from __future__ import annotations

from collections.abc import Sequence

import ipywidgets as widgets
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter

from analysis.models.instrument import PIXEL_KM2
from analysis.stats.instrument_sets import landings_per_set, timelines_per_set
from analysis.stats.tile import read_tile_track
from common.config import analysis_settings
from visualization import panels, wording
from visualization.config import GREY
from visualization.panels import Colour
from visualization.training_qualitative import config
from visualization.training_qualitative.picker import Coverage, tile_title


def observations(coverage: Coverage) -> widgets.Widget:
    """Draw one stacked panel per instrument set, over the whole tile."""
    tile_track = read_tile_track(coverage)
    window = tile_track.window if tile_track and tile_track.window.kept else None
    timeless = analysis_settings().criteria.timeless
    timelines = timelines_per_set(coverage)
    colours = panels.colours([timeline.label for timeline in timelines])
    figure, axes = stacked(
        len(timelines), 2.5 * len(timelines), sharex=True, sharey=True
    )
    for axis, timeline in zip(axes, timelines, strict=True):
        stems(axis, timeline.times, timeline.shares, colours[timeline.label])
        if not timeline.observed:
            note(axis, timeline.reason)
        axis.set_ylabel(timeline.label, rotation=0, ha="right", va="center", fontsize=9)
        tidy(axis, percent="y", grid="y")
        if window is None or timeline.iid in timeless:
            continue
        axis.axvspan(window.start, window.end, color="#9e9e9e", alpha=0.18, zorder=0)
        for edge in (window.start, window.end):
            axis.axvline(edge, zorder=4, **config.SURVEY)
    if window and any(timeline.iid not in timeless for timeline in timelines):
        marker = Line2D([], [], label="the window the tile earned", **config.SURVEY)
        axes[0].legend(handles=[marker], fontsize=8, loc="upper right", frameon=False)
    if not any(timeline.observed for timeline in timelines):
        axes[0].set_xlim(timelines[0].first, timelines[0].last)
    axes[0].set_ylim(-0.05, 1.05)
    panels.titled(axes[0], f"{tile_title(coverage)}  -  coverage per observation")
    axes[-1].set_xlabel("Observation start time")
    figure.supylabel("Share of the tile covered by one observation", fontsize=10)
    figure.tight_layout()
    return panels.rendered(figure)


def cumulative(coverage: Coverage) -> widgets.Widget:
    """Draw the running coverage of the whole tile, beside its total."""
    timelines = timelines_per_set(coverage)
    colours = panels.colours([timeline.label for timeline in timelines])
    figure = Figure(figsize=(13, 5))
    running, bars = figure.subplots(1, 2, width_ratios=[3, 1])
    last = timelines[0].last
    for timeline in timelines:
        if timeline.observed:
            times = [timeline.times[0], *timeline.times]
            fractions = [0.0, *timeline.running]
            if times[-1] < last:
                times.append(last)
                fractions.append(fractions[-1])
            style, label = "-", f"{timeline.covered:.1%}"
        else:
            times, fractions = [timeline.first, last], [0.0, 0.0]
            style, label = (0, (1, 3)), timeline.reason
        running.plot(
            times,
            fractions,
            linewidth=1.8,
            linestyle=style,
            color=colours[timeline.label],
            label=f"{timeline.label}  ({label})",
        )
    panels.titled(running, f"{tile_title(coverage)}  -  cumulative coverage")
    running.set_xlabel("Observation start time")
    running.set_ylabel("Share of the tile covered so far")
    running.set_ylim(0, 1.05)
    running.set_xlim(right=last)
    tidy(running, percent="y", grid="both")
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
    tidy(bars, percent="x", grid="x")
    figure.tight_layout()
    return panels.rendered(figure)


def pixels(coverage: Coverage) -> widgets.Widget:
    """Draw what each instrument lands on the tile, one observation at a time."""
    tile_track = read_tile_track(coverage)
    if tile_track is None:
        return panels.unavailable(config.NO_TRACK)
    landings = landings_per_set(tile_track)
    colours = panels.colours([landing.label for landing in landings])
    tall = 1.4 * len(landings) + config.TITLE_BAND
    figure, axes = stacked(len(landings), tall)
    for axis, landing in zip(axes, landings, strict=True):
        pixel_km2 = PIXEL_KM2[landing.iid]
        counts = np.asarray(landing.landed_km2, dtype=float) / pixel_km2
        bar = landing.bar_km2 / pixel_km2
        # The axis reaches the bar even where every look fell short of it
        top = max(float(counts.max()), bar) if counts.size else 0.0
        if top > 0.0:
            # A stem stands where the looks landing, as tall as there are of them
            counted, edges = np.histogram(counts, bins=60, range=(0.0, top))
            middles = (edges[:-1] + edges[1:]) / 2.0
            standing = counted > 0
            stems(axis, middles[standing], counted[standing], colours[landing.label])
            if bar > 0.0:
                axis.axvline(
                    bar, color="#1a1a1a", linestyle=(0, (4, 2)), linewidth=1.0, zorder=3
                )
            axis.set_title(
                f"{counts.size:,} observations  -  "
                f"middle one lands "
                f"{wording.pixels(float(np.median(counts)), landing.iid)}"
                f"  -  asked for {wording.pixels(bar, landing.iid)}",
                fontsize=8,
                color=GREY,
                loc="left",
            )
            axis.set_xlim(0.0, top * 1.05)
            axis.set_ylim(0.0, int(counted.max()) * 1.25)
            axis.xaxis.set_major_formatter(
                FuncFormatter(lambda value, _: wording.compact(value))
            )
            axis.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=3))
        else:
            note(axis, config.NOTHING_LANDED)
            axis.set_xticks([])
        axis.set_ylabel(landing.label, rotation=0, ha="right", va="center", fontsize=9)
        axis.tick_params(labelsize=8)
        tidy(axis, grid="both")
    axes[-1].set_xlabel("Pixels one observation lands on the tile")
    figure.supylabel("Observations", fontsize=10)
    # The strip is measured in inches, so it holds however many panels there are
    above = config.TITLE_BAND / tall
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 1.0 - above))
    figure.text(
        0.01,
        1.0 - above / 2.0,
        f"{tile_title(coverage)}  -  pixels per observation",
        fontsize=12,
        va="center",
    )
    return panels.rendered(figure)


def stacked(count: int, height: float, **shared) -> tuple[Figure, list[Axes]]:
    """Open a figure of one panel per instrument set, stacked.

    Args:
        count: How many panels to stack.
        height: The figure's height in inches.
        shared: What the panels share, as matplotlib's subplots takes it.

    Returns:
        figure: The figure.
        axes: Its panels, top to bottom.
    """
    figure = Figure(figsize=(11, height))
    return figure, list(figure.subplots(count, 1, squeeze=False, **shared)[:, 0])


def stems(axis: Axes, at: Sequence, heights: Sequence, colour: Colour) -> None:
    """Stand a stem at each place, as tall as its height, topped by a point.

    Args:
        axis: The panel drawn on.
        at: Where each stem stands along the horizontal axis.
        heights: How tall each stem stands.
        colour: The colour the stems and points are drawn in.
    """
    axis.vlines(at, 0.0, heights, color=colour, alpha=0.35, linewidth=0.7)
    axis.scatter(
        at, heights, s=12, alpha=0.65, color=colour, edgecolors="none", zorder=3
    )


def note(axis: Axes, text: str, colour: str = GREY, size: float = 9) -> None:
    """Write a line across the middle of a panel that has nothing to draw."""
    axis.text(
        0.5,
        0.5,
        text,
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=size,
        color=colour,
    )


def tidy(axis: Axes, grid: str, percent: str | None = None) -> None:
    """Grid an axis faintly and drop its outer frame, reading one side as percent.

    Args:
        axis: The axis to tidy.
        grid: Which of its sides are gridded, "x", "y" or "both".
        percent: Which side reads as percent, "x" or "y", or None for neither.
    """
    if percent is not None:
        target = axis.xaxis if percent == "x" else axis.yaxis
        target.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axis.grid(axis=grid, alpha=0.25, linewidth=0.5)
    axis.spines[["top", "right"]].set_visible(False)
