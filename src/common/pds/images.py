"""Reading the binary image a PDS label describes."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from common.pds import labels


def _scaled_values(values: np.ndarray, label: dict[str, str]) -> np.ndarray:
    """Return stored samples times the label's SCALING_FACTOR plus its OFFSET.

    Args:
        values: The samples as stored.
        label: The parsed label describing them.

    Returns:
        values: The samples in the label's unit, as stored when nothing scales them.
    """
    factor = float(label.get("SCALING_FACTOR", 1.0))
    offset = float(label.get("OFFSET", 0.0))
    if factor == 1.0 and offset == 0.0:
        return values
    return values.astype("f4") * factor + offset


def load_cube(image: Path) -> tuple[np.ndarray, dict[str, str]]:
    """Read a whole `.img` image and the `.lbl` label beside it.

    Args:
        image: The `.img` file, whose `.lbl` sits beside it.

    Returns:
        values: The samples as lines by samples by bands, in the label's unit.
        label: The parsed label.

    Raises:
        FileNotFoundError: When the image or its label is missing.
        KeyError: When it names a sample type this cannot read.
    """
    label = labels.load(image.with_suffix(".lbl"))
    lines, samples, bands, order, dtype = labels.image_layout(label)
    # Read exactly the cube, so any table written after it is left alone.
    flat = np.fromfile(image, dtype=dtype, count=lines * samples * bands)
    # BIL writes one line's bands together, BSQ whole bands one after another.
    if order == labels.BIL:
        held = flat.reshape(lines, bands, samples).transpose(0, 2, 1)
    else:
        held = flat.reshape(bands, lines, samples).transpose(1, 2, 0)
    return _scaled_values(held, label), label


def load_window(
    image: Path,
    label: dict[str, str],
    lines: tuple[int, int],
    samples: tuple[int, int],
) -> np.ndarray:
    """Read one rectangle of a single band `.img` image, seeking to its first line.

    Args:
        image: The `.img` file.
        label: The parsed label describing it.
        lines: The first line to read, and the line after the last.
        samples: The first sample to read, and the sample after the last.

    Returns:
        values: The samples inside those bounds, lines by samples, in the label's unit.

    Raises:
        KeyError: When it names a sample type this cannot read.
        ValueError: When the image holds more than the one band this reads.
    """
    _, line_samples, bands, _, dtype = labels.image_layout(label)
    if bands != 1:
        raise ValueError(f"{image.name} holds {bands} bands rather than one.")
    count = lines[1] - lines[0]
    with image.open("rb") as handle:
        handle.seek(lines[0] * line_samples * np.dtype(dtype).itemsize)
        flat = np.fromfile(handle, dtype=dtype, count=count * line_samples)
    held = flat.reshape(count, line_samples)[:, samples[0] : samples[1]]
    return _scaled_values(held, label)
