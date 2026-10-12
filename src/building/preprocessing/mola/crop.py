"""Cutting one MOLA grid to the tiles it was kept for, a polar cap read directly."""

from __future__ import annotations

import numpy as np

from building.preprocessing.common import cut, geometry
from building.preprocessing.common.models.position import Position
from building.preprocessing.mola import delay, projection, resample
from building.preprocessing.mola.merge_sheets import merge_sheets
from building.preprocessing.mola.models.observation import MolaObservation
from building.preprocessing.mola.models.sample import MolaSample
from common.maths.geodesy import TURN
from common.models.tile import Tile
from common.pds import images, labels


def mola_sample(
    source: Position, label: dict[str, str], elevation: np.ndarray, frame: Tile
) -> MolaSample:
    """Return the elevation one grid holds over a tile, on square ground samples.

    Args:
        source: Where the grid's lines and samples sit, in degrees or polar metres.
        label: What the products it was read from say about it, merged.
        elevation: The elevation above the areoid in metres, lines by samples.
        frame: The local frame of the tile it was kept for.

    Returns:
        sample: The crop, its elevations placed on the radargram's delay rows.
    """
    target = resample.square_position(frame)
    held, covered = resample.resampled_elevation(elevation, source, target)
    inside = (
        (target.north >= frame.min_lat)
        & (target.north <= frame.max_lat)
        & ((target.east - frame.west_lon) % TURN <= frame.span)
    )
    rows, reached = delay.radargram_rows(held)
    return MolaSample(
        position=cut.placed(target, frame),
        label=label,
        inside=geometry.partial_mask(inside),
        valid=geometry.partial_mask(covered),
        elevation=held,
        delay=rows,
        delay_inside=reached,
    )


def crop_polar(observation: MolaObservation, frame: Tile) -> MolaSample | None:
    """Return one polar grid resampled over its tile's box.

    Args:
        observation: The polar grid that landed, holding the one product it is.
        frame: The local frame of the tile it was kept for.

    Returns:
        sample: The grid resampled over that tile, or None where it reaches none of it.

    Raises:
        FileNotFoundError: When the grid or its label is missing.
        KeyError: When the label names a sample type this cannot read.
        ValueError: When it names a projection this cannot read.
    """
    (image,) = observation.files.values()
    label = labels.load(image.with_suffix(".lbl"))
    grid = projection.grid_position(label)
    held = cut.overlap(grid, frame)
    if held is None:
        return None
    lines, samples = held.bounds
    elevation = images.load_window(
        image,
        label,
        (int(lines[0]), int(lines[-1]) + 1),
        (int(samples[0]), int(samples[-1]) + 1),
    )
    window = Position(grid.north[lines], grid.east[samples], True, grid.grid)
    return mola_sample(window, labels.merge(label), elevation, frame)


def crop(observation: MolaObservation, frame: Tile) -> MolaSample | None:
    """Return one grid resampled over its tile's box, its sheets merged.

    Args:
        observation: The sheets of the grid that landed.
        frame: The local frame of the tile it was kept for.

    Returns:
        sample: The grid resampled over that tile, or None where a polar one reaches
            none of it.

    Raises:
        ValueError: When the sheets that landed leave part of its box unwritten.
    """
    if observation.grid.polar:
        return crop_polar(observation, frame)
    label, elevation, samples = merge_sheets(observation, frame)
    return mola_sample(samples, label, elevation, frame)
