"""Averaging one CTX crop over square blocks of pixels, its masks and grid with it."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from building.preprocessing.common import geometry
from building.preprocessing.ctx.models.sample import CtxSample

WITHIN = (1, 3)


def split_blocks(array: np.ndarray, factor: int) -> np.ndarray:
    """Return an array split into blocks of `factor` along every axis.

    Args:
        array: The array, cut from its first element on every axis.
        factor: How many elements a side one block holds.

    Returns:
        blocks: Each axis split into its blocks and the elements within one, every
            trailing element short of a whole block dropped.
    """
    kept = tuple(slice(0, size - size % factor) for size in array.shape)
    split = [held for size in array.shape for held in (size // factor, factor)]
    return array[kept].reshape(split)


def pooled_mask(mask: np.ndarray | None, factor: int) -> np.ndarray | None:
    """Return a mask over the blocks, set where any pixel of one is.

    Args:
        mask: The mask over the pixels, or None for all.
        factor: How many pixels a side one block holds.

    Returns:
        pooled: The mask over the blocks, or None where it marks every one.
    """
    if mask is None:
        return None
    return geometry.partial_mask(split_blocks(mask, factor).any(axis=WITHIN))


def pooled_sample(sample: CtxSample, factor: int) -> CtxSample:
    """Return one crop averaged over blocks of `factor` pixels a side.

    Args:
        sample: The crop, on the grid ISIS projected it onto.
        factor: How many pixels a side one block holds.

    Returns:
        pooled: The crop on the blocks, each the mean of its measured pixels or
            zero where none is, and placed at the mean of its pixels.

    Raises:
        ValueError: When the crop's grid does not hold one ground axis each.
    """
    position = sample.position
    if not position.separable:
        raise ValueError("Cannot pool a CTX crop whose grid is not separable.")
    north, east = (
        split_blocks(axis, factor).mean(axis=1)
        for axis in (position.north, position.east)
    )
    measured = sample.measured_ground
    held = split_blocks(measured, factor)
    image = np.where(held, split_blocks(sample.image, factor).astype(np.float32), 0)
    count = held.sum(axis=WITHIN, dtype=np.float32)
    label = sample.label
    return replace(
        sample,
        position=replace(position, north=north, east=east),
        label=label
        | {
            "PixelResolution": str(float(label["PixelResolution"]) * factor),
            "Scale": str(float(label["Scale"]) / factor),
        },
        inside=pooled_mask(sample.inside, factor),
        valid=None if sample.valid is None else pooled_mask(measured, factor),
        image=image.sum(axis=WITHIN) / np.maximum(count, 1),
    )
