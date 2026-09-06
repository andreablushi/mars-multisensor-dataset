"""What one cropped CRISM observation contributes to the dataset."""

from __future__ import annotations

from pathlib import Path

import utils.disk.paths as paths
from building.configs import crism as configs
from building.metadata.models.feature import FeatureFrame
from building.preprocessing.common import store
from building.preprocessing.crism.models.sample import CrismSample


def write(
    held: CrismSample, frame: FeatureFrame, root: Path = paths.DATASET_ROOT
) -> Path:
    """Write one cropped observation down, its cube and what each band holds.

    The detector is calibrated column by column, so a wavelength is written
    against the column it was measured on rather than against the band alone.

    Args:
        held: The observation cut to the feature it was kept for.
        frame: That feature's local frame.
        root: The dataset's own root directory.

    Returns:
        The directory it was written in.
    """
    _, sample, band = configs.LAYOUT.dims
    return store.write_sample(
        held,
        {
            configs.LAYOUT.measurement: (held.cube, configs.LAYOUT.dims),
            "wavelengths": (held.wavelengths, (sample, band)),
            "columns": (held.columns, (sample,)),
        },
        configs.LAYOUT,
        frame,
        root,
    )
