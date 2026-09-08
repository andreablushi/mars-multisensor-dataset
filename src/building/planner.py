"""Turning what a build could do into the products it still has to fetch and cut."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime
from itertools import chain, zip_longest
from pathlib import Path

import httpx

from analysis import dataset_list
from analysis.selector.models.selection import Selection
from building.dispatcher import INSTRUMENTS
from building.metadata.feature import feature_metadata
from building.models.job import Job, Plan
from building.models.settings import Settings
from building.preprocessing.common.store import sample_path
from shared.models.feature import Feature


def build_plan(
    settings: Settings,
    root: Path,
    ode: httpx.Client | None = None,
    *,
    force: bool = False,
) -> Plan:
    """Work out every product one build has to fetch, and what to cut it to.

    Args:
        settings: The settled choices for the build, which size it. Which
            instruments it covers is not among them.
        root: The directory this build of the dataset is written in.
        ode: The client an instrument searched by ground is looked up through,
            or None to leave those instruments out of the plan.
        force: When True, plan products every crop of which is already written.

    Returns:
        plan: The plan, its jobs heaviest first so no long one is picked up last.

    Raises:
        FileNotFoundError: When no selection has been written to build from.
    """
    picked, crowded = _sampled(dataset_list.read_dataset_list(), settings)
    features = [feature_metadata(one.feature) for one in picked]
    wanted: dict[tuple[str, str], list[Feature]] = defaultdict(list)
    taken: dict[tuple[str, str], datetime] = {}
    unread = 0
    for one, feature in zip(picked, features, strict=True):
        # A feature is built whole, every observation this build has an instrument for.
        for kept in one.observations:
            named = INSTRUMENTS.get(kept.iid)
            # Skip a product no instrument builds, and an id naming no observation.
            read = named.observation_id if named else None
            if read and (held := read(kept.pdsid)):
                # Both detectors name one observation, so it is cut from once
                if feature.frame not in wanted[(kept.iid, held)]:
                    wanted[(kept.iid, held)].append(feature.frame)
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
            for feature in features:
                # What it names is mosaicked to one box, so it is asked for alone.
                for held in named.identifiers(feature.frame, ode):
                    asked.append((name, held, (feature.frame,)))

    jobs, skipped = [], 0
    for instrument, identifier, held in asked:
        left = tuple(
            frame
            for frame in held
            if force or not sample_path(frame, instrument, identifier, root).exists()
        )
        skipped += len(held) - len(left)
        if left:
            when = taken.get((instrument, identifier))
            jobs.append(Job(instrument, identifier, left, when))
    return Plan(
        # Heaviest first, weighed by the product and not by how many features want it.
        jobs=tuple(
            sorted(
                jobs,
                key=lambda job: (
                    -INSTRUMENTS[job.instrument].worker_bytes,
                    -len(job.frames),
                ),
            )
        ),
        features=tuple(features),
        skipped_existing=skipped,
        unread=unread,
        crowded=crowded,
    )


def _sampled(
    picked: Sequence[Selection], settings: Settings
) -> tuple[list[Selection], int]:
    """Keep the share of the features one build covers, evenly across classes.

    Args:
        picked: What the search left of every feature it searched.
        settings: The settled choices for the build, whose cap is read before the
            draw, so a share is a share of what a build may cover.

    Returns:
        kept: The selections to build, in the order the selection was written.
        crowded: How many features were left out for holding too many observations.
    """
    passed = [one for one in picked if one.feature.kept]
    cap = settings.max_observations
    kept = [one for one in passed if len(one.observations) <= cap]
    crowded = len(passed) - len(kept)
    wanted = round(settings.share * len(kept))
    if wanted >= len(kept):
        return kept, crowded
    classes: dict[str, list[int]] = defaultdict(list)
    for at, one in enumerate(kept):
        classes[one.feature.feature_class].append(at)
    draw = random.Random(settings.seed)
    for held in classes.values():
        draw.shuffle(held)
    order = sorted(classes)
    draw.shuffle(order)
    # One from each class in turn, so every class is reached before any is drawn twice.
    rounds = zip_longest(*(classes[name] for name in order))
    taken = [at for at in chain.from_iterable(rounds) if at is not None]
    return [kept[at] for at in sorted(taken[:wanted])], crowded
