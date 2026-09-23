"""The oldest SHARAD track a tile keeps, cut to it as the build cuts it."""

from __future__ import annotations

import httpx
import ipywidgets as widgets
import numpy as np

from analysis.selector.models.selection import Selection
from analysis.visualization.common import panels
from building.configs import sharad as configs
from building.download import sharad as download
from building.metadata.tile import tile_metadata
from building.preprocessing.sharad import preprocess
from building.preprocessing.sharad.models.sample import SharadSample

FIGURE_SIZE = (5.0, 6.0)

ABOVE_ROWS = 100
BELOW_ROWS = 700
CONTRAST = (1.0, 99.5)

LOADING = "Fetching the SHARAD track..."
FAILED = "The SHARAD track could not be read: {reason}"
NO_TRACK = "The tile keeps no SHARAD track."
MISSED = "The track reaches none of the tile."


def plot(picked: Selection) -> widgets.Widget:
    """Show the oldest SHARAD track the tile keeps, cut to the tile."""
    identifier = next(
        (
            held
            for one in picked.observations
            if one.iid == configs.LAYOUT.instrument
            and (held := configs.NAMING.parse(one.pdsid))
        ),
        None,
    )
    if identifier is None:
        return panels.unavailable(NO_TRACK)
    frame = tile_metadata(picked.tile).frame

    def cut() -> SharadSample | None:
        """Bring the track into the build's cache and cut it to the tile.

        Returns:
            sample: The track cut to the tile, or None where it reaches none of it.
        """
        with httpx.Client() as client:
            download.fetch(identifier, client)
        return preprocess.crop(preprocess.read_observation(identifier), frame)

    return panels.loaded(cut, figure, LOADING, FAILED)


def figure(sample: SharadSample | None) -> widgets.Widget:
    """Draw the track's power in decibels, around the surface it sounded."""
    if sample is None:
        return panels.unavailable(MISSED)
    with np.errstate(divide="ignore", invalid="ignore"):
        power = 10.0 * np.log10(sample.power)
    power[~np.isfinite(power)] = np.nan
    # The brightest echo of a trace is taken as the surface it sounded
    surface = np.nanargmax(np.nan_to_num(power, nan=-np.inf), axis=0)
    top = max(int(surface.min()) - ABOVE_ROWS, 0)
    bottom = min(int(surface.max()) + BELOW_ROWS, power.shape[0])
    shown = power[top:bottom]
    low, high = np.nanpercentile(shown, CONTRAST)
    drawn, axis = panels.board(FIGURE_SIZE)
    step = configs.DELAY_INTERVAL_S * 1e6
    axis.imshow(
        shown,
        cmap="gray",
        vmin=low,
        vmax=high,
        aspect="auto",
        interpolation="nearest",
        extent=(0, shown.shape[1], bottom * step, top * step),
    )
    axis.set_xlabel("Trace")
    axis.set_ylabel("Delay (µs)")
    axis.tick_params(labelsize=8)
    axis.set_title(f"SHARAD {sample.identifier}", fontsize=12, loc="left")
    drawn.tight_layout()
    return panels.rendered(drawn)
