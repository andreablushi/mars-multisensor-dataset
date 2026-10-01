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
    if grid.product:
        products = {
            grid.product: configs.CACHE.files(identifier, grid.product, topography)
        }
    else:
        products = {
            directory.name: configs.CACHE.product_files(directory.name, topography)
            for directory in sorted(configs.CACHE.root.iterdir())
            if directory.is_dir()
            and configs.sheet_resolution(directory.name) == grid.resolution
        }
    images = {sheet: held[".img"] for sheet, held in products.items()}
    files = {sheet: image for sheet, image in images.items() if image.exists()}
    return MolaObservation(identifier, grid, files)
