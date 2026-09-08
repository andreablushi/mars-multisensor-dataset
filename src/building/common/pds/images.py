"""Reading the image a PDS label describes, whatever the product is."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from building.common.pds import labels


def build_cube(image: Path, label: dict[str, str]) -> np.ndarray:
    """Read one image into an array indexed by line, sample and band.

    Args:
        image: The `.img` file holding the values.
        label: The parsed label describing it.

    Returns:
        values: The values as lines by samples by bands, in the file's own band order
            and the label's unit.

    Raises:
        KeyError: When it names a sample type this cannot read.
    """
    # How the image is shaped, ordered and written.
    lines, samples, bands, stored, dtype = labels.layout(label)
    # How many values the cube holds, the trailing record excluded.
    wanted = lines * samples * bands
    # Read exactly those, so any table written after them is left alone.
    flat = np.fromfile(image, dtype=dtype, count=wanted)
    # BIL writes one line's bands together, so bands sit in the middle.
    if stored == labels.BIL:
        held = flat.reshape(lines, bands, samples).transpose(0, 2, 1)
    else:
        # BSQ writes whole bands one after another, so bands come first.
        held = flat.reshape(bands, lines, samples).transpose(1, 2, 0)
    return measured(held, label)


def measured(values: np.ndarray, label: dict[str, str]) -> np.ndarray:
    """Return what one image's stored values stand for.

    Args:
        values: The values as they were stored.
        label: The parsed label describing them.

    Returns:
        values: The values in the unit the label names, the stored ones where it asks
            for no scaling.
    """
    factor, offset = labels.scaling(label)
    if factor == 1.0 and offset == 0.0:
        return values
    return values.astype("f4") * factor + offset


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
        values: The values inside those bounds, as lines by samples, in the label's
            unit.

    Raises:
        KeyError: When it names a sample type this cannot read.
        ValueError: When the image holds more than the one band this reads.
    """
    _, across, bands, _, dtype = labels.layout(label)
    if bands != 1:
        raise ValueError(f"{image.name} holds {bands} bands rather than one.")
    with image.open("rb") as handle:
        handle.seek(lines[0] * across * np.dtype(dtype).itemsize)
        flat = np.fromfile(handle, dtype=dtype, count=(lines[1] - lines[0]) * across)
    held = flat.reshape(lines[1] - lines[0], across)[:, samples[0] : samples[1]]
    return measured(held, label)


def load_cube(image: Path) -> tuple[np.ndarray, dict[str, str]]:
    """Read one image and the label beside it that describes it.

    Args:
        image: The `.img` file holding the values, whose `.lbl` sits beside it.

    Returns:
        values: The values as lines by samples by bands, in the file's own band order.
        label: The parsed label describing them.

    Raises:
        FileNotFoundError: When the image or its label is missing.
        KeyError: When it names a sample type this cannot read.
    """
    label = labels.load(image.with_suffix(".lbl"))
    return build_cube(image, label), label


def load_plane(image: Path) -> tuple[np.ndarray, dict[str, str]]:
    """Read one single band image and the label beside it.

    Args:
        image: The `.img` file holding the values, whose `.lbl` sits beside it.

    Returns:
        values: The values as lines by samples.
        label: The parsed label describing them.

    Raises:
        FileNotFoundError: When the image or its label is missing.
        KeyError: When it names a sample type this cannot read.
    """
    cube, label = load_cube(image)
    return cube[:, :, 0], label
