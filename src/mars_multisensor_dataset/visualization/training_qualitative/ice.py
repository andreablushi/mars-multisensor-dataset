"""Two tiles side by side, one icy and one not, the SWIM grid over their mosaic."""

from __future__ import annotations

from functools import partial

import ipywidgets as widgets
import numpy as np

from mars_multisensor_dataset.analysis.ground_truth.swim.labels import read_ice_labels
from mars_multisensor_dataset.analysis.ground_truth.swim.models import IceLabel
from mars_multisensor_dataset.analysis.ground_truth.swim.sample import (
    GRID_POINTS,
    labels_positions,
)
from mars_multisensor_dataset.visualization import mosaic, panels
from mars_multisensor_dataset.visualization.placement import (
    PlacedTile,
    outlined_board,
    placed_tile,
)
from mars_multisensor_dataset.visualization.training_qualitative import config


def plot_ice() -> widgets.HBox:
    """Show each hardcoded tile with the ice SWIM finds between 1 and 5 m over it."""
    by_tile = {label.tile: label for label in read_ice_labels()}
    shown = []
    for title, tile in config.ICE_SAMPLES.items():
        placed = placed_tile(tile)
        space = mosaic.fetched(
            placed.crop(), partial(ice_map, placed, by_tile[tile], title)
        )
        space.layout = config.ICE_PANEL
        shown.append(space)
    return widgets.HBox(shown)


def ice_map(
    placed: PlacedTile,
    label: IceLabel,
    title: str,
    image: bytes,
) -> widgets.Image:
    """Draw one tile's crop of the mosaic with its SWIM grid laid over it.

    Args:
        placed: Where the tile falls in lon and lat.
        label: The ice SWIM finds under its grid.
        title: What the map is titled.
        image: The tile's crop, as the mosaic fetched it.

    Returns:
        map: The map, rendered.
    """
    lon, lat = labels_positions(
        np.array([label.centre_lon]), np.array([label.centre_lat])
    )
    side = (GRID_POINTS, GRID_POINTS)
    figure, axis = outlined_board(placed, (7.0, 6.0), image, zorder=3)
    shown = axis.pcolormesh(
        placed.around(lon).reshape(side),
        lat.reshape(side),
        np.reshape(label.consistency_1_5m, side),
        shading="nearest",
        cmap=config.ICE_CMAP,
        vmin=-1.0,
        vmax=1.0,
        alpha=config.ICE_ALPHA,
    )
    figure.colorbar(shown, ax=axis, shrink=0.8, label="Ice consistency, 1 to 5 m")
    depths = [
        name
        for name, icy in (("at 1 to 5 m", label.ice_1_5m), ("below 5 m", label.ice_5m))
        if icy
    ]
    verdict = f"icy {' and '.join(depths)}" if depths else "icy at neither depth"
    panels.titled(axis, f"{title}\n{label.tile}, {verdict}")
    figure.tight_layout()
    return panels.rendered(figure)
