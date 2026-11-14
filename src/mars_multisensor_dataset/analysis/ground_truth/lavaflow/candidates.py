"""The candidate crops: every track over every tile of a region a paper names."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from mars_multisensor_dataset.analysis.coverage.artifacts import read_set_events
from mars_multisensor_dataset.analysis.ground_truth.lavaflow.models import (
    LavaFlowLabel,
    LavaFlowSettings,
)
from mars_multisensor_dataset.analysis.models.instrument import InstrumentSet
from mars_multisensor_dataset.building.configs import sharad
from mars_multisensor_dataset.common.maths import box
from mars_multisensor_dataset.common.models.tile import Tile


def candidate_labels(
    tiles: Sequence[Tile],
    radargrams: InstrumentSet,
    settings: LavaFlowSettings,
    floor_km2: float,
) -> list[LavaFlowLabel]:
    """List every crop of a named track in its region, and the unnamed ones beside.

    Args:
        tiles: Every tile of the grid.
        radargrams: The instrument set the SHARAD radargrams are measured as.
        settings: The settled choices, holding every region and its tracks.
        floor_km2: The ground a track has to land on a tile to count.

    Returns:
        labels: One label per track and tile, region by region, the named tracks
            first, each track south to north.
    """
    boxes = box.bounds_boxes(tiles)
    labels = []
    for name, region in settings.regions.items():
        reached = [
            tiles[at] for at in box.touching(boxes, box.bounds_box(region)).nonzero()[0]
        ]
        latitude = {tile.name: tile.centre_lat for tile in reached}
        looks = [
            (sharad.NAMING.observation_id(event.pdsid), event)
            for event in read_set_events(radargrams, reached)
            if event.own_km2 >= floor_km2
        ]
        named = [look for look in looks if look[0] in region.tracks]
        crossed = {event.tile for _, event in named}
        beside = [
            look
            for look in looks
            if look[0] not in region.tracks and look[1].tile in crossed
        ]
        # The longest unnamed tracks first, so few of them run over many tiles
        length = Counter(track for track, _ in beside)
        beside.sort(key=lambda look: (-length[look[0]], -look[1].own_km2))
        taken = Counter()
        unnamed = []
        for track, event in beside:
            if taken[event.tile] < settings.unnamed_per_tile:
                taken[event.tile] += 1
                unnamed.append((track, event))
        for held, is_named in ((named, True), (unnamed, False)):
            labels += [
                LavaFlowLabel(
                    tile=event.tile,
                    track=track,
                    pdsid=event.pdsid,
                    t_start=event.t_start,
                    region=name,
                    source=region.source,
                    named=is_named,
                )
                for track, event in sorted(
                    held, key=lambda look: (look[0], latitude[look[1].tile])
                )
            ]
    return labels
