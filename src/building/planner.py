"""Turning what a build could do into the products it still has to fetch and cut."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import httpx

from analysis import dataset_list
from analysis.selector.models.selection import Selection
from building.dispatcher import INSTRUMENTS
from building.metadata.tile import tile_metadata
from building.models.job import Job, Plan
from building.models.settings import Settings
from building.preprocessing.common.store import sample_path
from shared.models.tile import Tile


def build_plan(
    settings: Settings,
    root: Path,
    ode: httpx.Client | None = None,
    *,
    force: bool = False,
    published: frozenset[str] = frozenset(),
) -> Plan:
    """Work out every product one build has to fetch, and what to cut it to.

    Args:
        settings: The settled choices for the build, which size it. Which
            instruments it covers is not among them.
        root: The directory this build of the dataset is written in.
        ode: The client an instrument searched by ground is looked up through,
            or None to leave those instruments out of the plan.
        force: When True, plan products every crop of which is already written.
        published: The crops that count as written although no longer on disk,
            by their path relative to the root.

    Returns:
        plan: The plan, its jobs heaviest first so no long one is picked up last.

    Raises:
        FileNotFoundError: When no selection has been written to build from.
    """
    picked = _sampled(dataset_list.read_dataset_list(), settings)
    tiles = [tile_metadata(one.tile) for one in picked]
    wanted: dict[tuple[str, str], list[Tile]] = defaultdict(list)
    taken: dict[tuple[str, str], datetime] = {}
    unread = 0
    for one, tile in zip(picked, tiles, strict=True):
        # A tile is built whole, every observation this build has an instrument for.
        for kept in one.observations:
            named = INSTRUMENTS.get(kept.iid)
            # Skip a product no instrument builds, and an id naming no observation.
            read = named.observation_id if named else None
            if read and (held := read(kept.pdsid)):
                # Both detectors name one observation, so it is cut from once
                if tile.frame not in wanted[(kept.iid, held)]:
                    wanted[(kept.iid, held)].append(tile.frame)
                taken.setdefault((kept.iid, held), kept.t_start)
            else:
                unread += 1
    asked = [
        (instrument, identifier, tuple(held))
        for (instrument, identifier), held in wanted.items()
    ]
    if ode is not None:
        # An instrument the selection cannot name is asked which products hold it.
        for name, named in INSTRUMENTS.items():
            if not named.identifiers:
                continue
            for tile in tiles:
                # What it names is mosaicked to one box, so it is asked for alone.
                for held in named.identifiers(tile.frame, ode):
                    asked.append((name, held, (tile.frame,)))

    jobs, skipped = [], 0
    for instrument, identifier, held in asked:
        left = tuple(
            frame
            for frame in held
            if force
            or not (
                str(crop := sample_path(frame, instrument, identifier, Path()))
                in published
                or (root / crop).exists()
            )
        )
        skipped += len(held) - len(left)
        if left:
            when = taken.get((instrument, identifier))
            jobs.append(Job(instrument, identifier, left, when))
    return Plan(
        # Heaviest first, weighed by the product and not by how many tiles want it.
        jobs=tuple(
            sorted(
                jobs,
                key=lambda job: (
                    -INSTRUMENTS[job.instrument].worker_bytes,
                    -len(job.frames),
                ),
            )
        ),
        tiles=tuple(tiles),
        skipped_existing=skipped,
        unread=unread,
    )


def _sampled(picked: Sequence[Selection], settings: Settings) -> list[Selection]:
    """Keep the share of the tiles one build covers, drawn at random.

    Args:
        picked: What the search left of every tile it searched.
        settings: The settled choices for the build, whose share settles how much
            of what the filter kept one build covers.

    Returns:
        kept: The selections to build, in the order the selection was written.
    """
    kept = [one for one in picked if one.tile.kept]
    wanted = round(settings.share * len(kept))
    if wanted >= len(kept):
        return kept
    taken = random.Random(settings.seed).sample(range(len(kept)), wanted)
    return [kept[at] for at in sorted(taken)]
