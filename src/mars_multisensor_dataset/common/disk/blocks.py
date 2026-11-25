"""Storing one image as blocks of lossy WebP, and reading back any window of it."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

BLOCKS = "blocks"
OFFSETS = "block_offsets"

FORMAT = "webp"

LEVELS = 255


def encoded_image(
    values: np.ndarray,
    low: float,
    high: float,
    block: int,
    quality: int,
    method: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Return one image as 8-bit WebP blocks, and what reads it back.

    Args:
        values: The image as lines by samples, every value from low to high.
        low: The value the lowest level stands for.
        high: The value the highest level stands for.
        block: How many pixels a side one block holds, the last ones cut short.
        quality: The WebP quality, 0 to 100.
        method: The WebP encoder effort, 0 to 6.

    Returns:
        arrays: Every block's bytes, row after row, and the offset each starts at.
        encoding: The format, the levels' range, the block side and the image's shape.
    """
    levels = np.round((np.clip(values, low, high) - low) / (high - low) * LEVELS)
    quantized = levels.astype(np.uint8)
    written = []
    for top in range(0, quantized.shape[0], block):
        for left in range(0, quantized.shape[1], block):
            held = io.BytesIO()
            Image.fromarray(quantized[top : top + block, left : left + block]).save(
                held, "WEBP", quality=quality, method=method
            )
            written.append(np.frombuffer(held.getvalue(), dtype=np.uint8))
    offsets = np.cumsum([0, *(len(one) for one in written)], dtype=np.int64)
    encoding = {
        "format": FORMAT,
        "low": float(low),
        "high": float(high),
        "block": block,
        "quality": quality,
        "method": method,
        "shape": list(quantized.shape),
    }
    return {BLOCKS: np.concatenate(written), OFFSETS: offsets}, encoding


def decoded_window(
    arrays: dict[str, np.ndarray],
    encoding: dict[str, object],
    window: tuple[slice, slice],
) -> np.ndarray:
    """Return one window of an image stored as WebP blocks, decoding only its blocks.

    Args:
        arrays: The blocks' bytes and offsets, as `encoded_image` wrote them.
        encoding: What `encoded_image` described the image with.
        window: The lines and then the samples to read, each with a start and stop.

    Returns:
        values: The window's values, as float32 between the levels' range.
    """
    block = int(encoding["block"])
    across = -(-int(encoding["shape"][1]) // block)
    lines, samples = window
    levels = np.empty(
        (lines.stop - lines.start, samples.stop - samples.start), np.uint8
    )
    data, offsets = arrays[BLOCKS], arrays[OFFSETS]
    for row in range(lines.start // block, -(-lines.stop // block)):
        for column in range(samples.start // block, -(-samples.stop // block)):
            at = row * across + column
            held = data[offsets[at] : offsets[at + 1]].tobytes()
            decoded = np.asarray(Image.open(io.BytesIO(held)).convert("L"))
            top, left = row * block, column * block
            first, last = max(lines.start, top), min(lines.stop, top + block)
            west, east = max(samples.start, left), min(samples.stop, left + block)
            levels[
                first - lines.start : last - lines.start,
                west - samples.start : east - samples.start,
            ] = decoded[first - top : last - top, west - left : east - left]
    low, high = float(encoding["low"]), float(encoding["high"])
    return low + levels.astype(np.float32) * np.float32((high - low) / LEVELS)
