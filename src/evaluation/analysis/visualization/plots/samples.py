"""One tile of each of two classes, set side by side instrument by instrument."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path

import ipywidgets as widgets
import numpy as np
from matplotlib.figure import Figure

from common.analysis.visualization.common import panels
from common.building.common.layout import DELAY
from common.building.metadata.observation import ObservationMetadata
from common.building.preprocessing.common.read import read_sample
from common.building.preprocessing.common.store import MEASURED
from evaluation.analysis.models.label import Label

PANEL_HEIGHT = 3.4

# The percentiles every panel is stretched between, so a faint echo still shows.
STRETCH = (1.0, 99.5)

# Samples per axis a crop is thinned to before it is drawn.
THINNED = 1024


def plot(
    root: Path,
    records: Sequence[ObservationMetadata],
    labels: Sequence[Label],
    tiles: Sequence[str],
) -> widgets.Widget:
    """Draw one crop of every instrument for each tile, one column per tile."""
    classes = {one.tile: one.label for one in labels}
    instruments = sorted({one.instrument for one in records if one.tile in tiles})
    if not instruments:
        return panels.unavailable("None of these tiles is built on disk.")
    figure = Figure(figsize=(panels.FIGURE_WIDTH, PANEL_HEIGHT * len(instruments)))
    axes = figure.subplots(len(instruments), len(tiles), squeeze=False)
    for row, instrument in enumerate(instruments):
        for column, tile in enumerate(tiles):
            axis = axes[row][column]
            axis.set_title(f"{classes[tile]}, {tile}, {instrument}", fontsize=9)
            held = [
                one
                for one in records
                if one.tile == tile and one.instrument == instrument
            ]
            if not held:
                axis.set_axis_off()
                panels.note(axis, "not observed")
                continue
            arrays, meta = read_sample(root / held[0].path)
            values = arrays[meta["measurement"]].astype(np.float32)
            measured = arrays[MEASURED].astype(bool)
            dims = meta["dims"][meta["measurement"]]
            ground = [dims.index(name) for name in meta["ground"]]
            if DELAY in meta["axes"]:
                # A radargram is drawn as it was sounded, delay down and in decibels
                values[:, ~measured] = np.nan
                values = 10.0 * np.log10(np.where(values > 0.0, values, np.nan))
                axis.set_xlabel("Trace")
                axis.set_ylabel("Delay")
            else:
                # An image is drawn in plan, every other axis of it averaged
                values = np.moveaxis(values, ground, range(len(ground)))
                with warnings.catch_warnings():
                    # Unmeasured bands and ground are skipped without a warning
                    warnings.simplefilter("ignore", RuntimeWarning)
                    values = np.nanmean(values.reshape(*measured.shape, -1), axis=-1)
                values[~measured] = np.nan
                axis.set_axis_off()
            thin = tuple(
                slice(None, None, -(-size // THINNED)) for size in values.shape
            )
            low, high = np.nanpercentile(values, STRETCH)
            axis.imshow(
                values[thin],
                cmap="gray",
                aspect="auto",
                interpolation="nearest",
                vmin=low,
                vmax=high,
            )
    figure.tight_layout()
    return panels.rendered(figure)
