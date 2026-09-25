"""Cutting one MOLA grid to the tiles it was kept for, a polar cap read directly."""

from __future__ import annotations

import numpy as np

from building.preprocessing.common import cut
from building.preprocessing.common.models.relative_position import (
    RelativePosition,
)
from building.preprocessing.mola import delay, projection
from building.preprocessing.mola.merge_sheets import merge_sheets
from building.preprocessing.mola.models.observation import MolaObservation
from building.preprocessing.mola.models.sample import MolaSample
from common.models.tile import Tile
from common.pds import images, labels


def mola_sample(
    observation: MolaObservation,
    position: RelativePosition,
    label: dict[str, str],
    height: np.ndarray,
    inside: np.ndarray | None = None,
) -> MolaSample:
    """Return the height one grid holds over a tile, with the delay rows it sets.

    Args:
        observation: The grid the height was read from, its crop stored under it.
        position: Where every bin sits relative to the tile centre.
        label: What the products it was read from say about it, merged.
        height: The height above the areoid in metres, lines by samples.
        inside: Which bins truly fall in the tile's box, or None for all.

    Returns:
        sample: The crop, its heights placed on the radargram's delay rows.
    """
    rows, reached = delay.radargram_rows(height)
    return MolaSample(
        identifier=observation.identifier,
        position=position,
        label=label,
        inside=inside,
        elevation=height,
        delay=rows,
        delay_inside=reached,
    )


def crop_polar(observation: MolaObservation, frame: Tile) -> MolaSample | None:
    """Return the bins of one polar grid its tile's box keeps.

    Args:
        observation: The polar grid that landed, holding the one product it is.
        frame: The local frame of the tile it is read for.

    Returns:
        sample: The height over that tile, or None where the grid reaches none of it.

    Raises:
        FileNotFoundError: When the grid or its label is missing.
        KeyError: When the label names a sample type this cannot read.
        ValueError: When it names a projection this cannot read.
    """
    (image,) = observation.files.values()
    label = labels.load(image.with_suffix(".lbl"))
    held = cut.overlap(projection.grid_samples(label), frame)
    if held is None:
        return None
    lines, samples = held.bounds
    height = images.load_window(
        image,
        label,
        (int(lines[0]), int(lines[-1]) + 1),
        (int(samples[0]), int(samples[-1]) + 1),
    )
    return mola_sample(
        observation, held.position, labels.merge(label), height, held.inside
    )


def crop(observation: MolaObservation, frame: Tile) -> MolaSample | None:
    """Return the bins of one grid its tile's box keeps, its sheets merged into one.

    Args:
        observation: The sheets of the grid that landed.
        frame: The local frame of the tile it was kept for.

    Returns:
        sample: The height over that tile, or None where a polar grid misses it.

    Raises:
        ValueError: When the sheets that landed leave part of its box unwritten.
    """
    if observation.polar:
        return crop_polar(observation, frame)
    label, height, samples = merge_sheets(observation, frame)
    return mola_sample(observation, cut.placed(samples, frame), label, height)
