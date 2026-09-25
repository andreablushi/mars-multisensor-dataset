"""Reading one MOLA grid off disk, which of its sheets landed but none of their bins."""

from __future__ import annotations

from building.configs import mola as configs
from building.preprocessing.mola.models.observation import MolaObservation


def read_observation(identifier: str) -> MolaObservation:
    """Read which sheets of one grid a tile could be merged from.

    Args:
        identifier: The grid as `configs.GRIDS` names it, its sheets already cached.

    Returns:
        observation: The sheets of it that landed, only their paths known yet.
    """
    grid = configs.GRIDS[identifier]
    topography = configs.Kind.TOPOGRAPHY
    files = {}
    if grid.product:
        image = configs.CACHE.files(identifier, grid.product, topography)[".img"]
        if image.exists():
            files[grid.product] = image
    else:
        for directory in sorted(configs.CACHE.root.iterdir()):
            if not directory.is_dir():
                continue
            if configs.sheet_resolution(directory.name) != grid.resolution:
                continue
            image = configs.CACHE.product_files(directory.name, topography)[".img"]
            if image.exists():
                files[directory.name] = image
    return MolaObservation(identifier, grid, files)
