"""Reading the image a PDS label describes, whatever the product is."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from common.pds import labels


def scaled(values: np.ndarray, label: dict[str, str]) -> np.ndarray:
    """Return what one image's stored values stand for.

    Args:
        values: The values as they were stored.
        label: The parsed label describing them.

    Returns:
        values: The values in the label's unit, stored ones where none is scaled.
    """
    factor = float(label.get("SCALING_FACTOR", 1.0))
    offset = float(label.get("OFFSET", 0.0))
    if factor == 1.0 and offset == 0.0:
        return values
    return values.astype("f4") * factor + offset


def load_cube(image: Path) -> tuple[np.ndarray, dict[str, str]]:
    """Read one image and the label beside it that describes it.

    Args:
        image: The `.img` file holding the values, whose `.lbl` sits beside it.

    Returns:
        values: The values as lines by samples by bands, in the label's unit.
        label: The parsed label describing them.

    Raises:
        FileNotFoundError: When the image or its label is missing.
        KeyError: When it names a sample type this cannot read.
    """
    label = labels.load(image.with_suffix(".lbl"))
    lines, samples, bands, stored, dtype = labels.image_layout(label)
    # Read exactly the cube, so any table written after it is left alone.
    flat = np.fromfile(image, dtype=dtype, count=lines * samples * bands)
    # BIL writes one line's bands together, BSQ whole bands one after another.
    if stored == labels.BIL:
        held = flat.reshape(lines, bands, samples).transpose(0, 2, 1)
    else:
        held = flat.reshape(bands, lines, samples).transpose(1, 2, 0)
    return scaled(held, label), label


def load_window(
    image: Path,
    label: dict[str, str],
    lines: tuple[int, int],
    samples: tuple[int, int],
) -> np.ndarray:
    """Read only the rectangle of one single band image the bounds ask for.

    Args:
        image: The `.img` file holding the values.
        label: The parsed label describing it.
        lines: The first line to read, and the line after the last.
        samples: The first sample to read, and the sample after the last.

    Returns:
        values: The values inside those bounds, lines by samples, in its unit.

    Raises:
        KeyError: When it names a sample type this cannot read.
        ValueError: When the image holds more than the one band this reads.
    """
    _, across, bands, _, dtype = labels.image_layout(label)
    if bands != 1:
        raise ValueError(f"{image.name} holds {bands} bands rather than one.")
    with image.open("rb") as handle:
        handle.seek(lines[0] * across * np.dtype(dtype).itemsize)
        flat = np.fromfile(handle, dtype=dtype, count=(lines[1] - lines[0]) * across)
    held = flat.reshape(lines[1] - lines[0], across)[:, samples[0] : samples[1]]
    return scaled(held, label)
