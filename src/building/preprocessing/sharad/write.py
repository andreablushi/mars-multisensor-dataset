"""What one cropped SHARAD track contributes to the dataset."""

from __future__ import annotations

from pathlib import Path

import utils.disk.paths as paths
from building.configs import sharad as configs
from building.metadata.models.feature import FeatureFrame
from building.preprocessing.common import store
from building.preprocessing.sharad.models.sample import SharadSample


def write(
    held: SharadSample, frame: FeatureFrame, root: Path = paths.DATASET_ROOT
) -> Path:
    """Write one cropped track down, its echoes and which columns they were.

    Args:
        held: The track cut to the feature it was kept for.
        frame: That feature's local frame.
        root: The dataset's own root directory.

    Returns:
        The directory it was written in.
    """
    (trace,) = configs.LAYOUT.ground
    return store.write_sample(
        held,
        {
            configs.LAYOUT.measurement: (held.power, configs.LAYOUT.dims),
            "traces": (held.traces, (trace,)),
        },
        configs.LAYOUT,
        frame,
        root,
    )
