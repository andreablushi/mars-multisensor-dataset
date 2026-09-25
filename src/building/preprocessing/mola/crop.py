"""Cutting one MOLA grid to a tile: a polar cap read directly, sheets merged first."""

from __future__ import annotations

import numpy as np

from building.preprocessing.common import cut
from building.preprocessing.common.models.relative_position import (
    RelativePosition,
)
from building.preprocessing.common.models.samples import Samples
from building.preprocessing.mola import delay, projection
from building.preprocessing.mola.merge_sheets import merge_sheets
from building.preprocessing.mola.models.observation import MolaObservation
from building.preprocessing.mola.models.sample import MolaSample
from common.models.tile import Tile
from common.pds import images, labels


def crop(grid: MolaObservation, frame: Tile) -> MolaSample | None:
    """Return the bins of one grid its tile's box keeps, merged into one.

    Args:
        grid: The sheets of the grid that landed.
        frame: The local frame of the tile they are merged for.

    Returns:
        sample: The height over that tile, or None where a polar grid misses it.

    Raises:
        ValueError: When the sheets that landed leave part of its box unwritten.
    """
    if grid.polar:
        return crop_polar(grid, frame)
    label, height, down, across = merge_sheets(grid, frame)
    position = cut.placed(Samples(down, across, True, None), frame)
    return mola_sample(grid, position, label, height)


def crop_polar(grid: MolaObservation, frame: Tile) -> MolaSample | None:
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
    down, across, polar = projection.grid_axes(label)
    held = cut.overlap(Samples(down, across, True, polar), frame)
    if held is None:
        return None
    lines, samples = held.bounds
    height = images.load_window(
        image,
        label,
        (int(lines[0]), int(lines[-1]) + 1),
        (int(samples[0]), int(samples[-1]) + 1),
    )
    return mola_sample(grid, held.position, labels.merge(label), height, held.inside)


def mola_sample(
    grid: MolaObservation,
    position: RelativePosition,
    label: dict[str, str],
    height: np.ndarray,
    inside: np.ndarray | None = None,
) -> MolaSample:
    """Return the height one grid holds over a tile, with the delay rows it sets.

    Args:
        grid: The grid the height was read from, which its crop is stored under.
        position: Where every bin sits relative to the tile centre.
        label: What the products it was read from say about it, merged.
        height: The height above the areoid in metres, lines by samples.
        inside: Which bins truly fall in the tile's box, or None for all.

    Returns:
        sample: The crop, its heights placed on the radargram's delay rows.
    """
    rows, reached = delay.radargram_rows(height)
    return MolaSample(
        identifier=grid.name,
        position=position,
        label=label,
        inside=inside,
        elevation=height,
        delay=rows,
        delay_inside=reached,
    )
