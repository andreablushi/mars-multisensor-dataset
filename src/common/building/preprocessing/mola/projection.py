"""Placing one MOLA grid on its own projection, and cutting a polar one to a tile."""

from __future__ import annotations

import numpy as np

from common.building.common.pds import images, labels
from common.building.preprocessing.common import geometry
from common.building.preprocessing.common.models.relative_position import PolarGrid
from common.building.preprocessing.common.models.samples import Samples
from common.building.preprocessing.mola import delay
from common.building.preprocessing.mola.models.grid import MolaGrid
from common.building.preprocessing.mola.models.sample import MolaSample
from common.maths import physics
from common.models.tile import Tile

# The two projections the gridded record is written in.
EQUATORIAL = "SIMPLE CYLINDRICAL"
POLAR = "POLAR STEREOGRAPHIC"


def grid_axes(
    label: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, PolarGrid | None]:
    """Return what places every line and every sample of one grid.

    Args:
        label: The parsed label of one product.

    Returns:
        down: What every line holds, the latitude of it or its northing.
        across: What every sample holds, the longitude of it or its easting.
        polar: The pole the two are measured on, and None for an equatorial grid.

    Raises:
        ValueError: When the label names a projection this cannot read.
    """
    named = label["MAP_PROJECTION_TYPE"]
    # How fine the grid is, which both projections count in bins to the degree.
    resolution = float(label["MAP_RESOLUTION"])
    lines, samples = int(label["LINES"]), int(label["LINE_SAMPLES"])
    if named == POLAR:
        # A cap is placed from its middle, in the stereographic metres MAP_SCALE names
        radius = float(label["A_AXIS_RADIUS"]) * physics.METRES_PER_KM
        down = np.radians((lines / 2.0 - 0.5 - np.arange(lines)) / resolution) * radius
        across = (
            np.radians((np.arange(samples) - samples / 2.0 + 0.5) / resolution) * radius
        )
        return down, across, (0.0, float(label["CENTER_LATITUDE"]) > 0.0, radius)
    if named != EQUATORIAL:
        raise ValueError(f"Cannot place a {named} grid.")
    # How many degrees one pixel spans, the same in both directions.
    step = 1.0 / resolution
    # The projection counts pixels from one, from the offset it puts its origin at.
    north = (
        float(label["CENTER_LATITUDE"])
        - (1.0 - float(label["LINE_PROJECTION_OFFSET"])) * step
    )
    west = (
        float(label["CENTER_LONGITUDE"])
        + (1.0 - float(label["SAMPLE_PROJECTION_OFFSET"])) * step
    )
    return (
        north - np.arange(lines) * step,
        west + np.arange(samples) * step,
        None,
    )


def crop_polar(grid: MolaGrid, frame: Tile) -> MolaSample | None:
    """Return the bins of one polar grid its tile's box keeps.

    Args:
        grid: The polar grid that landed, holding the one product it is published as.
        frame: The local frame of the tile it is read for.

    Returns:
        sample: The height over that tile, or None where the grid reaches none of it.

    Raises:
        FileNotFoundError: When the grid or its label is missing.
        KeyError: When the label names a sample type this cannot read.
        ValueError: When it names a projection this cannot read.
    """
    (image,) = grid.files.values()
    label = labels.load(image.with_suffix(".lbl"))
    down, across, polar = grid_axes(label)
    held = geometry.overlap(Samples(down, across, True, polar), frame)
    if held is None:
        return None
    lines, samples = held.bounds
    height = images.load_window(
        image,
        label,
        (int(lines[0]), int(lines[-1]) + 1),
        (int(samples[0]), int(samples[-1]) + 1),
    )
    rows, inside = delay.radargram_rows(height)
    return MolaSample(
        identifier=grid.name,
        position=held.position,
        label=labels.merge(label),
        inside=held.inside,
        elevation=height,
        delay=rows,
        delay_inside=inside,
    )
