"""How many pixels one observation lands on a tile, instrument by instrument."""

from __future__ import annotations

import ipywidgets as widgets
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator

from analysis.stats.instrument_sets import landings_per_set
from analysis.stats.tile import read_tile_track
from analysis.visualization import panels, wording
from analysis.visualization.panels import Coverage

TITLE_BAND = 0.42

_NOTHING = "nothing on this tile"


def plot(coverage: Coverage) -> widgets.Widget:
    """Draw what each instrument lands on the tile, one observation at a time."""
    tile_track = read_tile_track(coverage)
    if tile_track is None:
        return panels.unavailable(_NOTHING)
    landings = landings_per_set(tile_track)
    colours = panels.colours([landing.label for landing in landings])
    tall = 1.4 * len(landings) + TITLE_BAND
    figure, axes = panels.stacked(len(landings), tall)
    for axis, landing in zip(axes, landings, strict=True):
        counts = np.asarray(landing.counts, dtype=float)
        # The axis reaches the bar even where every look fell short of it
        top = max(float(counts.max()), landing.bar) if counts.size else 0.0
        if top > 0.0:
            # A stem stands where the looks landing, as tall as there are of them
            counted, edges = np.histogram(counts, bins=60, range=(0.0, top))
            middles = (edges[:-1] + edges[1:]) / 2.0
            standing = counted > 0
            panels.stems(
                axis, middles[standing], counted[standing], colours[landing.label]
            )
            if landing.bar > 0.0:
                axis.axvline(
                    landing.bar,
                    color="#1a1a1a",
                    linestyle=(0, (4, 2)),
                    linewidth=1.0,
                    zorder=3,
                )
            unit = "traces" if landing.iid == wording.SOUNDER else "px"
            axis.set_title(
                f"{counts.size:,} observations  -  "
                f"middle one lands "
                f"{wording.compact(float(np.median(counts)))} {unit}"
                f"  -  asked for {wording.compact(landing.bar)} {unit}",
                fontsize=8,
                color=panels.GREY,
                loc="left",
            )
            axis.set_xlim(0.0, top * 1.05)
            axis.set_ylim(0.0, int(counted.max()) * 1.25)
            axis.xaxis.set_major_formatter(
                FuncFormatter(lambda value, _: wording.compact(value))
            )
            axis.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=3))
        else:
            panels.note(axis, _NOTHING)
            axis.set_xticks([])
        axis.set_ylabel(landing.label, rotation=0, ha="right", va="center", fontsize=9)
        axis.tick_params(labelsize=8)
        panels.tidy(axis, grid="both")
    axes[-1].set_xlabel("Pixels one observation lands on the tile")
    figure.supylabel("Observations", fontsize=10)
    # The strip is measured in inches, so it holds however many panels there are
    above = TITLE_BAND / tall
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 1.0 - above))
    figure.text(
        0.01,
        1.0 - above / 2.0,
        f"{panels.title(coverage)}  -  pixels per observation",
        fontsize=12,
        va="center",
    )
    return panels.rendered(figure)
