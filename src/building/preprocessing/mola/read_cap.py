"""Reading the sector of one polar cap a feature's box stands on."""

from __future__ import annotations

from building.common.pds import images, labels
from building.models.feature import FeatureFrame
from building.preprocessing.common.crop import polar_overlap
from building.preprocessing.mola import projection
from building.preprocessing.mola.models.grid import MolaGrid
from building.preprocessing.mola.models.observation import MolaObservation


def read_cap(grid: MolaGrid, frame: FeatureFrame) -> MolaObservation | None:
    """Return the sector of one cap the feature's box stands on.

    Args:
        grid: The cap that landed, holding the one product it is published as.
        frame: The local frame of the feature the cap is read for.

    Returns:
        The observation holding that sector and no more of the cap, or None
        where the cap reaches none of the box.

    Raises:
        FileNotFoundError: When the cap or its label is missing.
        KeyError: When the label names a sample type this cannot read.
        ValueError: When it names a projection this cannot read, or the cap did
            not land at all.
    """
    if len(grid.files) != 1:
        raise ValueError(f"{grid.name} is one product, and {len(grid.files)} landed.")
    (image,) = grid.files.values()
    label = labels.load(image.with_suffix(".lbl"))
    down, across, pole = projection.load(label)
    if pole is None:
        raise ValueError(f"{grid.name} is written on no pole.")
    held = polar_overlap(down, across, pole, frame)
    if held is None:
        return None
    lines, samples = held.bounds
    return MolaObservation(
        grid.name,
        label,
        images.load_window(
            image,
            label,
            (int(lines[0]), int(lines[-1]) + 1),
            (int(samples[0]), int(samples[-1]) + 1),
        ),
        down[lines],
        across[samples],
        pole,
    )
