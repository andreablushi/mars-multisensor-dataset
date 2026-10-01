"""The tile itself: the ground it covers, and what the search made of it."""

from __future__ import annotations

from functools import partial
from html import escape

import ipywidgets as widgets

from analysis.stats.artifacts import read_tile_selection
from analysis.stats.tile import read_tile_track
from analysis.utils.tile_group import tile_grid
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
    name = coverage[0].summary.tile
    placed = placed_tile(name)
    if placed is None:
        return panels.unavailable(mosaic.NO_BOX)
    title = panels.title(coverage)
    tile_track = read_tile_track(coverage)
    ground = f"{wording.area(tile_track.window.area_km2)}, " if tile_track else ""
    report = panels.tile_report(tile_grid().tile_named(name), read_tile_selection(name))
    criteria = analysis_settings().criteria
    asked = ["What the filter asks"]
    asked += [
        "  "
        + " or ".join(
            f"{iid} {share:.0%} of the ground" for iid, share in constraint.items()
        )
        for constraint in criteria.constraints
    ]
    asked += [
        f"  {iid} counts at {wording.pixels(pixels, iid)} a look"
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
        f"<b>{escape(title)}</b><br>{escape(ground + report)}"
        + "".join(
            f"<pre style='margin: 8px 0 0; line-height: 1.4'>{escape(text)}</pre>"
            for text in (observed, "\n".join(asked))
        ),
        layout=widgets.Layout(flex="0 0 360px"),
    )
    return widgets.HBox(
        [
            report,
            mosaic.fetched(placed.crop(), partial(tile_map, placed, title)),
        ],
        layout=widgets.Layout(
            align_items="flex-start", flex_flow="row nowrap", grid_gap="24px"
        ),
    )


def tile_map(placed: PlacedTile, title: str, image: bytes) -> widgets.Image:
    """Draw the tile's crop of the mosaic, with the ground it covers outlined.

    Args:
        placed: Where the tile falls in lon and lat.
        title: What the map is titled.
        image: The tile's crop, as the mosaic fetched it.

    Returns:
        map: The map, rendered.
    """
    drawn, axis = outlined_board(placed, (7.0, 6.0), image, zorder=3)
    panels.titled(axis, title)
    drawn.tight_layout()
    return panels.rendered(drawn)
