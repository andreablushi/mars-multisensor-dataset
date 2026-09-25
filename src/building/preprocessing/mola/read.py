"""Reading which sheets of one MOLA grid landed, before any of their bins."""

from __future__ import annotations

from building.configs import mola as configs
from building.preprocessing.mola.models.observation import MolaObservation


def read_observation(grid: str) -> MolaObservation:
    """Read which sheets of one grid a tile could be merged from.

    Args:
        grid: The grid as `configs.GRIDS` names it, its sheets already cached.

    Returns:
        grid: The sheets of it that landed, only their labels read yet.
    """
    held = configs.GRIDS[grid]
    files = {}
    if held.product:
        image = configs.CACHE.files(grid, held.product, configs.Kind.TOPOGRAPHY)[".img"]
        if image.exists():
            files[held.product] = image
    else:
        for directory in sorted(configs.CACHE.root.iterdir()):
            parts = configs.NAMING.parts(directory.name) if directory.is_dir() else None
            if not parts or configs.RESOLUTIONS[parts["step"]] != held.resolution:
                continue
            image = configs.CACHE.files(
                directory.name,
                configs.NAMING.product(directory.name, configs.Kind.TOPOGRAPHY),
                configs.Kind.TOPOGRAPHY,
            )[".img"]
            if image.exists():
                files[directory.name] = image
    return MolaObservation(grid, held.resolution, files, held.north is not None)
