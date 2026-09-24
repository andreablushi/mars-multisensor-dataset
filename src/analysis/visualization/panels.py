"""How a panel reads: its colours, its axes, its tables, and what stands in for it."""

from __future__ import annotations

import io
from collections.abc import Sequence
from html import escape
from itertools import cycle

import ipywidgets as widgets
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from analysis.coverage.models.coverage import SetCoverage

# The colour one instrument set is drawn in
Colour = tuple[float, float, float]

# The tile every panel is drawn for
Coverage = list[SetCoverage]

# One row of a written table
Row = Sequence[str]

GREY = "#8a8a8a"


def colours(labels: Sequence[str]) -> dict[str, Colour]:
    """Assign a colour to each instrument set."""
    return dict(zip(labels, cycle(plt.cm.tab10.colors), strict=False))


def board(size: tuple[float, float], projection: object = None) -> tuple[Figure, Axes]:
    """Open a figure off pyplot's registry, so a thread may draw on it."""
    figure = Figure(figsize=size)
    return figure, figure.subplots(subplot_kw={"projection": projection})


def stacked(count: int, height: float, **shared) -> tuple[Figure, list[Axes]]:
    """Open a figure of one panel per instrument set, stacked."""
    figure = Figure(figsize=(11, height))
    axes = figure.subplots(count, 1, squeeze=False, **shared)
    return figure, [axis for row in axes for axis in row]


def key_beside(figure: Figure, handles: Sequence) -> None:
    """Set a key beside a map, in a strip left clear down its right side."""
    figure.tight_layout(rect=(0.0, 0.0, 0.78, 1.0))
    figure.legend(
        handles=handles,
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(0.80, 0.92),
        frameon=False,
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
    """Grid an axis faintly and drop its outer frame, reading one side as percent."""
    if percent is not None:
        target = axis.xaxis if percent == "x" else axis.yaxis
        target.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axis.grid(axis=grid, alpha=0.25, linewidth=0.5)
    axis.spines[["top", "right"]].set_visible(False)


def rendered(figure: Figure) -> widgets.Image:
    """Turn a finished figure into a widget that can replace an earlier one."""
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=figure.dpi)
    return widgets.Image(
        value=buffer.getvalue(),
        format="png",
        layout=widgets.Layout(max_width="100%", height="auto"),
    )


def written(title: str, headings: Sequence[str], rows: Sequence[Row]) -> widgets.HTML:
    """Write a table out as a panel."""
    frame = pd.DataFrame(rows, columns=list(headings))
    return widgets.HTML(f"<b>{escape(title)}</b>{frame.to_html(index=False, border=0)}")


def unavailable(
    message: str = "Confirm a tile with local data above to fill this in.",
) -> widgets.HTML:
    """Build the grey panel shown when there is nothing to draw."""
    return widgets.HTML(
        f"""<div style="
            background: repeating-linear-gradient(45deg,
                #ebebeb, #ebebeb 10px, #e0e0e0 10px, #e0e0e0 20px);
            border: 1px solid #c4c4c4; border-radius: 6px; color: {GREY};
            font-family: sans-serif; padding: 28px; text-align: center;">
          <div style="font-size: 15px; font-weight: 600;">No local data</div>
          <div style="font-size: 13px; margin-top: 6px;">{escape(message)}</div>
        </div>"""
    )


def title(coverage: Coverage) -> str:
    """Return the tile a loaded coverage belongs to."""
    return f"Tile {coverage[0].summary.tile}"
