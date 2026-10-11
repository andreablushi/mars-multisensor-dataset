"""The ground of the tile on show: where it lies, and what the search left on it."""

from __future__ import annotations

from functools import partial
from html import escape

import ipywidgets as widgets

from analysis.stats.artifacts import read_tile_selection
from analysis.stats.order import config_rank
from analysis.stats.tile import (
    ground_by_instrument_count,
    measure_tile,
    read_tile_track,
)
from analysis.utils.tile_group import tile_grid
from common.config import analysis_settings
from visualization import mosaic, panels, wording
from visualization.config import NO_BOX
from visualization.panels import Row
from visualization.placement import placed_tile, tile_map
from visualization.training_qualitative import config
from visualization.training_qualitative.picker import Coverage, tile_report, tile_title


def basemap(coverage: Coverage) -> widgets.Widget:
    """Show the ground the tile covers, and what the filter asked of it."""
    name = coverage[0].summary.tile
    placed = placed_tile(name)
    if placed is None:
        return panels.unavailable(NO_BOX)
    title = tile_title(coverage)
    tile_track = read_tile_track(coverage)
    ground = f"{wording.area(tile_track.window.area_km2)}, " if tile_track else ""
    report = tile_report(tile_grid().tile_named(name), read_tile_selection(name))
    criteria = analysis_settings().criteria
    asked = [
        "What the filter asks",
        *(
            "  "
            + " or ".join(
                f"{iid} {share:.0%} of the ground" for iid, share in constraint.items()
            )
            for constraint in criteria.constraints
        ),
        *(
            f"  {iid} counts at {wording.pixels(pixels, iid)} a look"
            for iid, pixels in criteria.admits.items()
        ),
        f"  a window turns {criteria.span_ls:.0f} degrees of Mars' year at most",
        *(f"  {iid} counts whenever it came" for iid in sorted(criteria.timeless)),
    ]
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
        [report, mosaic.fetched(placed.crop(), partial(tile_map, placed, title))],
        layout=widgets.Layout(
            align_items="flex-start", flex_flow="row nowrap", grid_gap="24px"
        ),
    )


def window(coverage: Coverage) -> widgets.Widget:
    """Summarise what the search left on the tile on show."""
    tile_track = read_tile_track(coverage)
    if tile_track is None:
        return panels.unavailable(config.NO_TRACK)
    stats = measure_tile(tile_track)
    earned = tile_track.window
    # A window it earned always says when it opened, so only a refusal reads as none
    lasted = (
        f"{wording.duration(earned.days)}, "
        f"{earned.start:%Y-%m-%d} to {earned.end:%Y-%m-%d}"
        if earned.kept
        else "-"
    )
    rows: list[Row] = [
        ("Ground the tile covers", wording.area(earned.area_km2)),
        ("How long its window lasts", lasted),
    ]
    for iid, reach in sorted(stats.reached.items(), key=lambda by: config_rank(by[0])):
        taken = counted(reach.observations_taken, "observation")
        rows += [
            (
                f"Ground {iid} reaches",
                f"{reach.km2 / earned.area_km2:.0%}, "
                f"{wording.pixels_of_ground(reach.landed_km2, iid)}, from {taken}",
            ),
            (
                f"Mean pixels per {iid} observation",
                wording.pixels_of_ground(reach.landed_km2_per_look, iid),
            ),
        ]
    rows += [
        (
            f"Ground reached by {counted(shared, 'instrument')}",
            f"{wording.area(km2)}, {km2 / earned.area_km2:.0%}",
        )
        for shared, km2 in ground_by_instrument_count(stats.overlaps).items()
    ]
    return panels.written(
        f"{tile_title(coverage)}  -  what it holds",
        ("On this tile", "What it holds"),
        rows,
    )


def counted(number: float, noun: str) -> str:
    """Write how many of something there are, the noun made plural to match."""
    return f"{number:,.0f} {noun}" + ("" if number == 1 else "s")
