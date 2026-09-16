"""The tile on show, and everything the search left on it."""

from __future__ import annotations

import ipywidgets as widgets

from analysis.stats.tile import measure, read
from analysis.visualization.common import panels, quantities, tables, wording
from analysis.visualization.common.models.coverage import Coverage
from analysis.visualization.common.models.tables import Row

_HEADINGS = ("On this tile", "What it holds")
_NOTHING = "No instrument set filled a cell of this tile."
_NONE = "-"


def plot(coverage: Coverage) -> widgets.Widget:
    """Summarise what the search left on the tile on show."""
    if not coverage:
        return panels.unavailable()
    looks = read.read_tile(coverage)
    if looks is None:
        return panels.unavailable(_NOTHING)
    stats = measure.measured_tile(looks)
    window = looks.window
    # A window it earned always says when it opened, so only a refusal reads as none
    lasted = (
        f"{quantities.duration(window.days)}, "
        f"{window.start:%Y-%m-%d} to {window.end:%Y-%m-%d}"
        if window.kept
        else _NONE
    )
    rows: list[Row] = [
        ("Ground the tile covers", quantities.area(window.area_km2)),
        ("How long its window lasts", lasted),
    ]
    for iid in sorted(stats.reached):
        reach = stats.reached[iid]
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
            wording.ground(km2, window.area_km2),
        )
        for shared, km2 in measure.ground_by_instrument_count(stats.overlaps).items()
    ]
    return tables.written(
        f"{panels.title(coverage)}  -  what it holds", _HEADINGS, rows
    )
