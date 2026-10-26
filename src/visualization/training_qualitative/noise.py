"""Each SHARAD track's noise, measured and gated the way a build does it."""

from __future__ import annotations

from collections.abc import Sequence

import ipywidgets as widgets
import numpy as np
from matplotlib.figure import Figure

from analysis import paths
from analysis.utils.tile_group import tile_grid
from building.preprocessing.common.models.sample import Sample
from building.preprocessing.sharad.denoise import measured_noise
from building.preprocessing.sharad.models.sample import SharadSample
from building.preprocessing.sharad.normalize import radargram_db, surface_aligned
from common.disk.slugify import slugify
from common.models.tile import Tile
from visualization import panels
from visualization.training_qualitative import config, product
from visualization.training_qualitative.preprocessing import cached_panels, shown_crop


def plot_sharad_noise() -> widgets.VBox:
    """Show each SHARAD noise sample aligned, preprocessed, and profiled."""
    grid = tile_grid()
    return widgets.VBox(
        [
            cached_panels(
                paths.PANELS_ROOT
                / slugify("SHARAD")
                / f"noise_{identifier}_{tile}_{product.sharad_gate_name()}.json",
                "SHARAD",
                (identifier,),
                grid.tile_named(tile),
                noise_images,
            )
            for identifier, tile in config.SHARAD_NOISE_SAMPLES
        ]
    )


def aligned_radargram(identifier: str, sample: SharadSample) -> np.ndarray:
    """Return a crop's radargram aligned on its surface, before its gate.

    Args:
        identifier: The track, its files already in the download cache.
        sample: The crop the build would write of it.

    Returns:
        aligned: The radargram in dB over the crop's surface, delay rows by the
            valid traces the gate is measured on.
    """
    cropped = radargram_db(product.sharad_power(identifier)[:, sample.traces])
    return surface_aligned(cropped, sample.measured_ground)[:, sample.valid]


def noise_images(
    name: str, identifiers: Sequence[str], frame: Tile
) -> tuple[list[widgets.Widget], list[Sample]]:
    """Draw one SHARAD track aligned, preprocessed and profiled, then its noise.

    Args:
        name: The instrument, as the build names it.
        identifiers: The track alone, its files already in the download cache.
        frame: The tile it is cut to.

    Returns:
        rows: The radargram aligned and preprocessed beside its mean delay profile,
            in one figure, then what its noise measures.
        samples: The crop the build would write.
    """
    (identifier,) = identifiers
    observation, sample = product.observed_sample(name, identifier, frame)
    aligned = aligned_radargram(identifier, sample)
    noise = measured_noise(aligned, observation.gate)
    preprocessed = sample.power[:, sample.valid]
    zenith = float(np.median(sample.incidence_deg[sample.valid]))
    figure = Figure(figsize=(17, 7))
    figure.suptitle(
        f"{identifier} on {frame.name}, solar zenith {zenith:.0f}°", fontsize=14
    )
    axes = figure.subplots(1, 3, width_ratios=(1, 1, 1.1))
    for axis, (text, values) in zip(
        axes,
        (
            ("Aligned (dB over surface)", aligned),
            ("Preprocessed (dB over surface)", preprocessed),
        ),
        strict=False,
    ):
        shown_crop(figure, axis, values, (values.min(), values.max()), text)
        axis.set_xlabel("Trace")
    profile = 10 * np.log10((10 ** (noise.averaged / 10)).mean(axis=1))
    surface = int(np.median(noise.averaged.argmax(axis=0)))
    axis = axes[-1]
    axis.plot(profile, np.arange(len(profile)), color="k", linewidth=0.8)
    axis.axvline(
        noise.threshold,
        color="tab:red",
        label=f"threshold, false alarm {observation.gate.false_alarm:g}",
    )
    axis.axhspan(
        0,
        surface - observation.gate.guard_rows,
        color="tab:blue",
        alpha=0.15,
        label="noise above",
    )
    axis.axhspan(
        noise.deep, len(profile), color="tab:green", alpha=0.15, label="noise below"
    )
    axis.set_ylim(len(profile), 0)
    axis.legend(loc="lower left")
    axis.set_xlabel("dB over surface")
    panels.titled(axis, f"Mean delay profile, {observation.gate.looks} traces")
    for axis in axes:
        axis.set_ylabel("Delay row")
    figure.tight_layout()
    measured = (
        f"{noise.mean:.2f}",
        f"{noise.spread:.2f}",
        f"{noise.threshold - noise.mean:.2f}",
        f"{(noise.averaged >= noise.threshold).mean():.3f}",
        f"{(noise.averaged[~noise.mask] > noise.threshold).mean():.3f}",
    )
    return [
        panels.rendered(figure),
        panels.written("Noise", config.NOISE_STATISTICS, [measured]),
    ], [sample]
