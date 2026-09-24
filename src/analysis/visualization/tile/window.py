"""The window the tile on show earned, and the ground it reaches, as a table."""

from __future__ import annotations

import ipywidgets as widgets

from analysis.stats.tile import (
    ground_by_instrument_count,
    measure_tile,
    read_tile_track,
)
from analysis.visualization import panels, wording
from analysis.visualization.panels import Coverage, Row


def plot(coverage: Coverage) -> widgets.Widget:
    """Summarise what the search left on the tile on show."""
    tile_track = read_tile_track(coverage)
    if tile_track is None:
        return panels.unavailable("No instrument set filled a cell of this tile.")
    stats = measure_tile(tile_track)
    window = tile_track.window
    # A window it earned always says when it opened, so only a refusal reads as none
    lasted = (
        f"{wording.duration(window.days)}, "
        f"{window.start:%Y-%m-%d} to {window.end:%Y-%m-%d}"
        if window.kept
        else "-"
    )
    rows: list[Row] = [
        ("Ground the tile covers", wording.area(window.area_km2)),
        ("How long its window lasts", lasted),
    ]
    for iid, reach in sorted(stats.reached.items()):
        taken = wording.counted(reach.observations_taken, "observation")
        rows += [
            (
                f"Ground {iid} reaches",
                f"{reach.km2 / window.area_km2:.0%}, "
                f"{wording.pixels(reach.pixels)}, from {taken}",
            ),
            (
                f"Mean pixels per {iid} observation",
                wording.pixels(reach.pixels_per_look),
            ),
        ]
    rows += [
        (
            f"Ground reached by {wording.counted(shared, 'instrument')}",
            (
                f"{wording.area(km2)}, {km2 / window.area_km2:.0%}"
                if km2
                else wording.NOTHING
            ),
        )
        for shared, km2 in ground_by_instrument_count(stats.overlaps).items()
    ]
    return panels.written(
        f"{panels.title(coverage)}  -  what it holds",
        ("On this tile", "What it holds"),
        rows,
    )
