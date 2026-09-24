"""The tile itself: the ground it covers, and what the search made of it."""

from __future__ import annotations

from html import escape

import ipywidgets as widgets

from analysis.stats.tile import read
from analysis.visualization.common import mosaic, panels, wording
from analysis.visualization.common.models.box import Box
from analysis.visualization.common.models.coverage import Coverage
from analysis.visualization.tile.models.placing import Placed
from analysis.visualization.tile.plots import placing
from common.config import analysis_settings

MAP_FIGURE_SIZE = (7.0, 6.0)
REPORT_WIDTH = "360px"

TILE_EDGE = "#ffffff"
TILE_WIDTH = 1.4

WINDOW_FOUND = "There is a window that covers the tile"
WINDOW_MISSING = "No window available that respects the requirements"
ASKED_HEADING = "What the filter asks"


def plot(coverage: Coverage) -> widgets.Widget:
    """Show the ground the tile covers, and what the filter asked of it."""
    if not coverage:
        return panels.unavailable()
    summary = coverage[0].summary
    looks = read.read_tile(coverage)
    grid = placing.placed(summary.tile)
    if grid is None:
        return panels.unavailable(mosaic.BASEMAP_FAILED.format(reason=mosaic.NO_BOX))
    title = panels.title(coverage)
    box = grid.box()
    verdict = WINDOW_FOUND if looks and looks.window.kept else WINDOW_MISSING
    criteria = analysis_settings().window
    asked = "\n".join(
        [ASKED_HEADING]
        + [
            "  "
            + " or ".join(
                f"{iid} {share:.0%} of the ground" for iid, share in constraint.items()
            )
            for constraint in criteria.constraints
        ]
        + [
            f"  {iid} counts at {wording.pixels(pixels)} a look"
            for iid, pixels in criteria.admits.items()
        ]
        + [f"  a window turns {criteria.span_ls:.0f} degrees of Mars' year at most"]
        + [f"  {iid} counts whenever it came" for iid in sorted(criteria.timeless)]
    )
    body = "\n".join(
        f"{instrument.label:16s} {instrument.summary.n_obs:6,d} observations"
        f"{f'  ({instrument.reason})' if instrument.reason else ''}"
        for instrument in coverage
    )
    report = widgets.HTML(
        f"<b>{escape(title)}</b><br>"
        f"{summary.tile_area_km2:,.1f} km2 bounding box, "
        f"{box.south:.3f} to {box.north:.3f} lat, "
        f"{box.west:.3f} to {box.east:.3f} lon<br>"
        f"{escape(verdict)}"
        f"<pre style='margin: 8px 0 0; line-height: 1.4'>{escape(body)}</pre>"
        f"<pre style='margin: 8px 0 0; line-height: 1.4'>{escape(asked)}</pre>",
        layout=widgets.Layout(flex=f"0 0 {REPORT_WIDTH}"),
    )
    return widgets.HBox(
        [
            report,
            mosaic.fetched(box, lambda image: figure(grid, box, image, title)),
        ],
        layout=widgets.Layout(
            align_items="flex-start", flex_flow="row nowrap", grid_gap="24px"
        ),
    )


def figure(grid: Placed, box: Box, image: bytes, title: str) -> widgets.Widget:
    """Draw the tile's crop of the mosaic, with the ground it covers outlined."""
    drawn, axis = panels.board(MAP_FIGURE_SIZE)
    mosaic.draw(axis, box, image)
    lon, lat = grid.outline()
    axis.plot(lon, lat, color=TILE_EDGE, linewidth=TILE_WIDTH, zorder=3)
    axis.set_title(title, fontsize=12, loc="left")
    drawn.tight_layout()
    return panels.rendered(drawn)
