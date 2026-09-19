"""Reading the sheets that landed and cutting them to the tile they are merged for."""

from __future__ import annotations

from building.configs import mola as configs
from building.preprocessing.common import geometry
from building.preprocessing.common.models.samples import Samples
from building.preprocessing.mola import delay, projection
from building.preprocessing.mola.merge_sheets import merge_sheets
from building.preprocessing.mola.models.grid import MolaGrid
from building.preprocessing.mola.models.sample import MolaSample
from common.models.tile import Tile


def read_observation(grid: str) -> MolaGrid:
    """Read which sheets of one grid a tile could be merged from.

    Args:
        grid: The grid, as `configs.GRIDS` names it, whose sheets must already
            be in the cache that `download.fetch` puts them in.

    Returns:
        grid: The sheets of it that landed, no more than a label of which is read
            until a box says which bins to take.
    """
    held = configs.GRIDS[grid]
    files = {}
    if held.product:
        image = configs.CACHE.files(grid, held.product, configs.TOPOGRAPHY)[".img"]
        if image.exists():
            files[held.product] = image
    else:
        for directory in sorted(configs.CACHE.root.iterdir()):
            parts = configs.NAMING.parts(directory.name) if directory.is_dir() else None
            if not parts or configs.RESOLUTIONS[parts["step"]] != held.resolution:
                continue
            image = configs.CACHE.files(
                directory.name,
                configs.NAMING.product(directory.name, configs.TOPOGRAPHY),
                configs.TOPOGRAPHY,
            )[".img"]
            if image.exists():
                files[directory.name] = image
    return MolaGrid(grid, held.resolution, files, held.north is not None)


def crop(grid: MolaGrid, frame: Tile) -> MolaSample | None:
    """Return the bins of one grid its tile's box keeps, merged into one.

    Args:
        grid: The sheets of the grid that landed.
        frame: The local frame of the tile they are merged for.

    Returns:
        sample: The height over that tile, or None where a polar grid misses it.

    Raises:
        ValueError: When the sheets that landed leave part of its box unwritten.
    """
    if grid.polar:
        return projection.crop_polar(grid, frame)
    observation = merge_sheets(grid, frame)
    rows, inside = delay.radargram_rows(observation.topography)
    return MolaSample(
        identifier=observation.identifier,
        position=geometry.placed(
            Samples(observation.down, observation.across, observation.separable, None),
            frame,
        ),
        label=observation.label,
        elevation=observation.topography,
        delay=rows,
        delay_inside=inside,
    )
