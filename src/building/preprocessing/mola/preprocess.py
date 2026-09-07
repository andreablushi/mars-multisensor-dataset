"""Reading the tiles that landed and cutting them to the feature they are merged for."""

from __future__ import annotations

from building.configs import mola as configs
from building.models.feature import FeatureFrame
from building.preprocessing.common.crop import overlap, taken
from building.preprocessing.mola.merge_tiles import merge_tiles
from building.preprocessing.mola.models.grid import MolaGrid
from building.preprocessing.mola.models.sample import MolaSample


def read_observation(grid: str) -> MolaGrid:
    """Read which tiles of one grid a feature could be merged from.

    Args:
        grid: The grid, as `configs.GRIDS` names it, whose tiles must already
            be in the cache that `download.fetch` puts them in.

    Returns:
        The tiles of it that landed, which no more than a label of is read
        until a feature's own box says which bins of them to take.
    """
    resolution = configs.GRIDS[grid]
    files = {}
    for directory in sorted(configs.CACHE.root.iterdir()):
        parts = configs.NAMING.parts(directory.name) if directory.is_dir() else None
        if not parts or configs.RESOLUTIONS[parts["step"]] != resolution:
            continue
        image = configs.CACHE.files(
            directory.name,
            configs.NAMING.product(directory.name, configs.TOPOGRAPHY),
            configs.TOPOGRAPHY,
        )[".img"]
        if image.exists():
            files[directory.name] = image
    return MolaGrid(grid, resolution, files)


def crop(grid: MolaGrid, frame: FeatureFrame) -> MolaSample | None:
    """Return the bins of one grid its feature's box keeps, merged into one.

    Args:
        grid: The tiles of the grid that landed.
        frame: The local frame of the feature they are merged for.

    Returns:
        The height over that feature, or None where the grid reaches none of it.

    Raises:
        ValueError: When the tiles that landed leave part of its box unwritten.
    """
    observation = merge_tiles(grid, frame)
    held = overlap(observation, frame)
    if held is None:
        return None
    return MolaSample(
        identifier=observation.identifier,
        position=held.position,
        label=observation.label,
        inside=held.inside,
        topography=taken(observation.topography, held.bounds),
    )
