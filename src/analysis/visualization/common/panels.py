"""How a figure reads: its colours, its axes, and what stands in for it."""

from __future__ import annotations

import io
import threading
from collections.abc import Callable, Sequence
from datetime import datetime
from html import escape
from itertools import cycle

import ipywidgets as widgets
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from analysis.visualization.common.models.colours import Colour
from analysis.visualization.common.models.coverage import Coverage

GREY = "#8a8a8a"

PLACEHOLDER = "320px"

FIGURE_WIDTH = 11

KEY_WIDTH = 0.78
KEY_SIDE = 0.80
KEY_TOP = 0.92

SURVEY_LINE = "#1a1a1a"
SURVEY_STYLE = (0, (6, 3))
SURVEY_WIDTH = 0.8
SURVEY_SHADE = "#9e9e9e"
SURVEY_ALPHA = 0.18


def colours(labels: Sequence[str]) -> dict[str, Colour]:
    """Assign a colour to each instrument set."""
    return dict(zip(labels, cycle(plt.cm.tab10.colors), strict=False))


def board(size: tuple[float, float], projection: object = None) -> tuple[Figure, Axes]:
    """Open a figure off pyplot's registry, so a thread may draw on it."""
    figure = Figure(figsize=size)
    return figure, figure.subplots(subplot_kw={"projection": projection})


def stacked(count: int, height: float, **shared) -> tuple[Figure, list[Axes]]:
    """Open a figure of one panel per instrument set, stacked."""
    figure = Figure(figsize=(FIGURE_WIDTH, height))
    axes = figure.subplots(count, 1, squeeze=False, **shared)
    return figure, [axis for row in axes for axis in row]


def key_beside(figure: Figure, handles: Sequence) -> None:
    """Set a key beside a map, in a strip left clear down its right side."""
    figure.tight_layout(rect=(0.0, 0.0, KEY_WIDTH, 1.0))
    figure.legend(
        handles=handles,
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(KEY_SIDE, KEY_TOP),
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


def tidy(axis: Axes, percent: str, grid: str) -> None:
    """Format an axis as percentages, grid it faintly, and drop its outer frame."""
    target = axis.xaxis if percent == "x" else axis.yaxis
    target.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axis.grid(axis=grid, alpha=0.25, linewidth=0.5)
    axis.spines[["top", "right"]].set_visible(False)


def shade(axis: Axes, open_for: Sequence[tuple[datetime, datetime]]) -> None:
    """Mark the stretches of time the windows are open over."""
    for opened, closed in open_for:
        axis.axvspan(opened, closed, color=SURVEY_SHADE, alpha=SURVEY_ALPHA, zorder=0)
        for edge in (opened, closed):
            axis.axvline(
                edge,
                color=SURVEY_LINE,
                linestyle=SURVEY_STYLE,
                linewidth=SURVEY_WIDTH,
                zorder=4,
            )


def rendered(figure: Figure) -> widgets.Image:
    """Turn a finished figure into a widget that can replace an earlier one."""
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=figure.dpi)
    return widgets.Image(
        value=buffer.getvalue(),
        format="png",
        layout=widgets.Layout(max_width="100%", height="auto"),
    )


def loaded[T](
    fetch: Callable[[], T],
    draw: Callable[[T], widgets.Widget],
    waiting: str,
    failed: str,
) -> widgets.Box:
    """Claim the space one figure goes in and fill it off the thread that fetches it."""
    space = widgets.Box(
        [
            widgets.HTML(
                f"<div style='width: {PLACEHOLDER}; height: {PLACEHOLDER};"
                f" display: flex; align-items: center; justify-content: center;"
                f" box-sizing: border-box; padding: 10px; text-align: center;"
                f" background: #f2f2f2; border: 1px solid #d8d8d8;"
                f" border-radius: 4px; color: {GREY};"
                f" font-family: sans-serif; font-size: 12px;'>"
                f"{escape(waiting)}</div>"
            )
        ]
    )

    def fill() -> None:
        """Fetch what is drawn and put the figure in the claimed space."""
        try:
            held = fetch()
        except Exception as exc:
            space.children = (unavailable(failed.format(reason=exc)),)
            return
        space.children = (draw(held),)

    threading.Thread(target=fill, daemon=True).start()
    return space


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
