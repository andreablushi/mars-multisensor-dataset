"""What one cropped MOLA tile contributes to the dataset."""

from __future__ import annotations

from pathlib import Path

import utils.disk.paths as paths
from building.configs import mola as configs
from building.metadata.models.feature import FeatureFrame
from building.preprocessing.common import store
from building.preprocessing.mola.models.sample import MolaSample


def write(
    held: MolaSample, frame: FeatureFrame, root: Path = paths.DATASET_ROOT
) -> Path:
    """Write one cropped tile down, its height and how each bin was measured.

    Args:
        held: The tile cut to the feature it was kept for.
        frame: That feature's local frame.
        root: The dataset's own root directory.

    Returns:
        The directory it was written in.
    """
    return store.write_sample(
        held,
        {
            configs.LAYOUT.measurement: (held.topography, configs.LAYOUT.dims),
            "counts": (held.counts, configs.LAYOUT.dims),
        },
        configs.LAYOUT,
        frame,
        root,
    )
