"""One MOLA grid as it comes off disk, its sheets found but none of their bins read."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

from mars_multisensor_dataset.building.configs.mola import Grid
from mars_multisensor_dataset.building.preprocessing.common.models.position import (
    Position,
)


@dataclass(frozen=True, slots=True)
class MolaObservation:
    """One grid, before any of its bins are read.

    Attributes:
        identifier: The grid as `configs.GRIDS` names it.
        grid: How fine it is, and whether it is projected onto a pole.
        files: The image of every product that landed, by sheet or polar product.
    """

    identifier: str
    grid: Grid
    files: dict[str, Path]


class MergedSheets(NamedTuple):
    """The one grid every sheet a tile stands on writes its part of.

    Attributes:
        label: What the sheets that write part of the box say about it, merged.
        elevation: The elevation above the areoid in metres over the box, lines by
            samples.
        position: The latitude of every line, falling southward, and the longitude of
            every sample, rising eastward past a turn.
    """

    label: dict[str, str]
    elevation: np.ndarray
    position: Position
