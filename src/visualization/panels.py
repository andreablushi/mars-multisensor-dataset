"""How a panel reads: its colours, title, key, tables, and what stands in for it."""

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

from visualization import config

# The colour one instrument set is drawn in
Colour = tuple[float, float, float]

# One row of a written table
Row = Sequence[str]


def colours(labels: Sequence[str]) -> dict[str, Colour]:
    """Assign a colour to each label."""
    return dict(zip(labels, cycle(plt.cm.tab10.colors), strict=False))


def titled(axis: Axes, text: str) -> None:
    """Title a panel the way every panel is titled."""
    axis.set_title(text, fontsize=12, loc="left")


def key_beside(figure: Figure, handles: Sequence) -> None:
    """Set a key beside a map, in a strip left clear down its right side.

    Args:
        figure: The figure the map is drawn on.
        handles: What the key lists.
    """
    figure.tight_layout(rect=(0.0, 0.0, 0.78, 1.0))
    figure.legend(
        handles=handles,
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(0.80, 0.92),
        frameon=False,
    )


def rendered(figure: Figure) -> widgets.Image:
    """Turn a finished figure into a widget that can replace an earlier one.

    Args:
        figure: The finished figure.

    Returns:
        image: The figure as a PNG widget.
    """
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=figure.dpi)
    return png_widget(buffer.getvalue())


def png_widget(value: bytes) -> widgets.Image:
    """Return PNG bytes as a widget that fits its area's width."""
    return widgets.Image(
        value=value,
        format="png",
        layout=widgets.Layout(max_width="100%", height="auto"),
    )


def written(title: str, headings: Sequence[str], rows: Sequence[Row]) -> widgets.HTML:
    """Write a table out as a panel.

    Args:
        title: What the table is titled.
        headings: The heading of each column.
        rows: The rows, one string per column.

    Returns:
        table: The titled table, as HTML.
    """
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
            border: 1px solid #c4c4c4; border-radius: 6px; color: {config.GREY};
            font-family: sans-serif; padding: 28px; text-align: center;">
          <div style="font-size: 15px; font-weight: 600;">No local data</div>
          <div style="font-size: 13px; margin-top: 6px;">{escape(message)}</div>
        </div>"""
    )
