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
        The values as lines by samples by bands, in the band order the file
        stores them in.

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
        return flat.reshape(lines, bands, samples).transpose(0, 2, 1)
    # BSQ writes whole bands one after another, so bands come first.
    return flat.reshape(bands, lines, samples).transpose(1, 2, 0)


def load_cube(image: Path) -> tuple[np.ndarray, dict[str, str]]:
    """Read one image and the label beside it that describes it.

    Args:
        image: The `.img` file holding the values, whose `.lbl` sits beside it.

    Returns:
        The values as lines by samples by bands, in the band order the file
        stores them in, and the parsed label describing them.

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
        The values as lines by samples, and the parsed label describing them.

    Raises:
        FileNotFoundError: When the image or its label is missing.
        KeyError: When it names a sample type this cannot read.
    """
    cube, label = load_cube(image)
    return cube[:, :, 0], label
