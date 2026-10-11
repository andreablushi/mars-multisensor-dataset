"""SHARAD and its cluttergram as taken, beside the crop the model receives."""

from __future__ import annotations

import ipywidgets as widgets
import numpy as np
from matplotlib.figure import Figure

from analysis.visualization import panels
from analysis.visualization.preprocessing import product
from building.configs import sharad as configs
from common.models.tile import Tile
from common.pds import images

NAME = "SHARAD"


def drawn_panel(identifier: str, frame: Tile) -> widgets.VBox:
    """Show one SHARAD track and its cluttergram whole, beside the model input."""
    product.fetch_product(NAME, identifier, frame)
    observation = product.product_observation(NAME, identifier)
    sample = product.tile_sample(NAME, observation, frame)
    power, _ = images.load_cube(
        configs.CACHE.product_files(identifier, configs.Kind.OBSERVATION)[".img"]
    )
    figure = Figure(figsize=(11, 7))
    axes = figure.subplot_mosaic(
        [["radargram", "crop"], ["cluttergram", "crop"]], width_ratios=(2, 1)
    )
    for name, values, text in (
        ("radargram", panels.stretched(power[:, :, 0]), "Downloaded sample"),
        (
            "cluttergram",
            panels.stretched(np.asarray(observation.clutter)),
            "Downloaded cluttergram",
        ),
        (
            "crop",
            product.scaled_input(product.model_input(NAME, sample)),
            "Model input",
        ),
    ):
        axis = axes[name]
        axis.imshow(
            values,
            cmap="gray",
            aspect="auto",
            interpolation="nearest",
            vmin=0,
            vmax=1,
        )
        panels.titled(axis, text)
        axis.set_xlabel("Trace")
        axis.set_ylabel("Delay row")
    for name in ("radargram", "cluttergram"):
        for edge in (sample.traces[0], sample.traces[-1]):
            axes[name].axvline(edge, color="tab:orange", linewidth=1.0)
    figure.tight_layout()
    return widgets.VBox(
        [
            panels.rendered(figure),
            product.acquisition_table(NAME, identifier, sample, frame),
        ]
    )


def plot(identifier: str, frame: Tile) -> widgets.VBox:
    """Show one SHARAD track, drawn once, then read off disk."""
    return product.kept_panel(NAME, identifier, frame, drawn_panel)
