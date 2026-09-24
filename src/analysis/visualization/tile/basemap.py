"""The tile itself: the ground it covers, and what the search made of it."""

from __future__ import annotations

from html import escape

import ipywidgets as widgets

from analysis.stats.tile import read_tile_track
from analysis.visualization import mosaic, panels, wording
from analysis.visualization.panels import Coverage
from analysis.visualization.tile.placement import (
    PlacedTile,
    outlined_board,
    placed_tile,
)
from common.config import analysis_settings


def plot(coverage: Coverage) -> widgets.Widget:
    """Show the ground the tile covers, and what the filter asked of it."""
    summary = coverage[0].summary
    tile_track = read_tile_track(coverage)
    placed = placed_tile(summary.tile)
    if placed is None:
        return panels.unavailable(mosaic.NO_BOX)
    title = panels.title(coverage)
    box = placed.box()
    if tile_track and tile_track.window.kept:
        verdict = "There is a window that covers the tile"
    else:
        verdict = "No window available that respects the requirements"
    criteria = analysis_settings().window
    asked = ["What the filter asks"]
    asked += [
        "  "
        + " or ".join(
            f"{iid} {share:.0%} of the ground" for iid, share in constraint.items()
        )
        for constraint in criteria.constraints
    ]
    asked += [
        f"  {iid} counts at {wording.pixels(pixels)} a look"
        for iid, pixels in criteria.admits.items()
    ]
    asked.append(
        f"  a window turns {criteria.span_ls:.0f} degrees of Mars' year at most"
    )
    asked += [f"  {iid} counts whenever it came" for iid in sorted(criteria.timeless)]
    observed = "\n".join(
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
        + "".join(
            f"<pre style='margin: 8px 0 0; line-height: 1.4'>{escape(text)}</pre>"
            for text in (observed, "\n".join(asked))
        ),
        layout=widgets.Layout(flex="0 0 360px"),
    )
    return widgets.HBox(
        [
            report,
            mosaic.fetched(box, lambda image: tile_map(placed, image, title)),
        ],
        layout=widgets.Layout(
            align_items="flex-start", flex_flow="row nowrap", grid_gap="24px"
        ),
    )


def tile_map(placed: PlacedTile, image: bytes, title: str) -> widgets.Image:
    """Draw the tile's crop of the mosaic, with the ground it covers outlined.

    Args:
        placed: Where the tile falls in lon and lat.
        image: The tile's crop, as the mosaic fetched it.
        title: What the map is titled.

    Returns:
        map: The map, rendered.
    """
    drawn, axis = outlined_board(placed, (7.0, 6.0), image, zorder=3)
    axis.set_title(title, fontsize=12, loc="left")
    drawn.tight_layout()
    return panels.rendered(drawn)
