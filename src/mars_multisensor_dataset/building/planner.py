"""Turning what a build could do into the products it still has to fetch and cut."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Sequence
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from mars_multisensor_dataset.analysis.selector.models.selection import Selection
from mars_multisensor_dataset.building.metadata.tile import tile_metadata
from mars_multisensor_dataset.building.models.instrument import (
    INSTRUMENTS,
    observation_identifier,
)
from mars_multisensor_dataset.building.models.job import Job, Plan
from mars_multisensor_dataset.building.preprocessing.common.store import sample_path
from mars_multisensor_dataset.common.models.tile import Tile


def build_plan(
    picked: Sequence[Selection],
    root: Path,
    *,
    force: bool,
    published: frozenset[str],
    instruments: Collection[str] | None = None,
) -> Plan:
    """Work out every product one build has to fetch, and what to cut it to.

    Args:
        picked: The tiles to build, each with the observations its window keeps.
        root: The directory this build of the dataset is written in.
        force: When True, plan products every crop of which is already written.
        published: The crops counted as written though off disk, by relative path.
        instruments: The instruments built, or None for every one.

    Returns:
        plan: The plan, its jobs heaviest first so no long one is picked up last.
    """
    built = INSTRUMENTS.keys() if instruments is None else instruments
    tiles = [tile_metadata(one.tile) for one in picked]
    wanted: dict[tuple[str, str], list[Tile]] = defaultdict(list)
    taken: dict[tuple[str, str], datetime] = {}
    unread = 0
    for one, tile in zip(picked, tiles, strict=True):
        # A tile is built whole, every observation this build has an instrument for.
        for kept in one.observations:
            observation = kept.iid in built and observation_identifier(
                kept.iid, kept.pdsid
            )
            # Skip a product no instrument builds, and an id naming no observation.
            if not observation:
                unread += 1
                continue
            product = (kept.iid, observation)
            # Both detectors name one observation, so it is cut from once
            if tile.frame not in wanted[product]:
                wanted[product].append(tile.frame)
            taken.setdefault(product, kept.t_start)
    asked = [
        Job(instrument, identifier, tuple(frames), taken[(instrument, identifier)])
        for (instrument, identifier), frames in wanted.items()
    ]
    # An instrument the selection cannot name is given the grid each tile falls on.
    for name, named in INSTRUMENTS.items():
        if not named.grid_of or name not in built:
            continue
        for tile in tiles:
            # What it names is mosaicked to one box, so it is asked for alone.
            asked.append(Job(name, named.grid_of(tile.frame), (tile.frame,), None))

    jobs, skipped = [], 0
    for job in asked:
        left = []
        for frame in job.frames:
            crop = sample_path(frame, job.instrument, job.identifier)
            if not force and (str(crop) in published or (root / crop).exists()):
                skipped += 1
            else:
                left.append(frame)
        if left:
            jobs.append(replace(job, frames=tuple(left)))
    # Heaviest first, weighed by the product and not by how many tiles want it.
    jobs.sort(
        key=lambda job: (-INSTRUMENTS[job.instrument].worker_bytes, -len(job.frames))
    )
    return Plan(tuple(jobs), tuple(tiles), skipped=skipped, unread=unread)
