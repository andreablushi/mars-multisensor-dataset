"""What the dataset says about one tile, and where it sits on Mars."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from common.analysis.selector.models.selection import SelectedTile
from common.disk import parquet
from common.models.tile import Tile


@dataclass(frozen=True, slots=True)
class TileMetadata:
    """One tile of the dataset: when it was seen, and where it lies.

    Attributes:
        frame: The local frame every observation of it is placed against, which
            is the only place its absolute position is written down.
        centre_lon: The longitude that frame is centred on, which is the centre
            every crop of the tile was cut against.
        centre_lat: The latitude it is centred on, for the same reason.
        area_km2: How much ground the tile's box covers.
        kept: Whether the filter gave the tile a place at all.
        window_start: When the earliest observation it keeps was taken, or None
            where it earned no window.
        window_end: When the latest one was taken, or None for the same reason.
        window_days: How long that window runs.
        window_share: The insisted shares rooted together, as a share of the
            tile.
        observations_kept: How many observations the filter left it.
    """

    frame: Tile
    centre_lon: float
    centre_lat: float
    area_km2: float
    kept: bool
    window_start: datetime | None
    window_end: datetime | None
    window_days: float
    window_share: float
    observations_kept: int

    @property
    def identity(self) -> str:
        """Return what tells this tile from every other.

        Returns:
            identity: Its name.
        """
        return self.frame.name


def tile_metadata(tile: SelectedTile) -> TileMetadata:
    """Return what the dataset holds about one tile the selection kept.

    Args:
        tile: The tile's own row, as the selection wrote it.

    Returns:
        metadata: The metadata, its frame carrying the tile's box.
    """
    frame = Tile(
        band=tile.band,
        column=tile.column,
        min_lat=tile.min_lat,
        max_lat=tile.max_lat,
        west_lon=tile.west_lon,
        east_lon=tile.east_lon,
    )
    return TileMetadata(
        frame=frame,
        centre_lon=frame.centre_lon,
        centre_lat=frame.centre_lat,
        area_km2=tile.area_km2,
        kept=tile.kept,
        window_start=tile.start,
        window_end=tile.end,
        window_days=tile.days,
        window_share=tile.geo_mean,
        observations_kept=tile.taken,
    )


SCHEMA = parquet.schema_of(TileMetadata)
