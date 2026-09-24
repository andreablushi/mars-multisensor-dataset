"""The mark the per-observation panels share: a stem topped by a point."""

from __future__ import annotations

from collections.abc import Sequence

from matplotlib.axes import Axes

from analysis.visualization.panels import Colour


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
