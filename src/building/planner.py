"""Turning what a build could do into the products it still has to fetch and cut."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime
from itertools import chain, zip_longest
from pathlib import Path

import httpx

import utils.disk.paths as paths
from analysis import dataset_list
from analysis.selector.models.selection import Selection
from building.dispatcher import INSTRUMENTS
from building.metadata.feature import feature_metadata
from building.models.feature import FeatureFrame
from building.models.job import Job, Plan
from building.models.settings import Settings
from building.preprocessing.common.store import sample_path


def build_plan(
    settings: Settings,
    ode: httpx.Client | None = None,
    root: Path = paths.DATASET_ROOT,
    *,
    force: bool = False,
) -> Plan:
    """Work out every product one build has to fetch, and what to cut it to.

    Args:
        settings: The settled choices for the build, which size it.
        ode: The client an instrument searched by ground is looked up through,
            or None to leave those instruments out of the plan.
        root: The dataset's own root directory.
        force: When True, plan products every crop of which is already written.

    Returns:
        The plan, its jobs heaviest first so no long one is picked up last.

    Raises:
        FileNotFoundError: When no selection has been written to build from.
    """
    picked = _sampled(dataset_list.read_dataset_list(), settings)
    features = [feature_metadata(one.feature) for one in picked]
    wanted: dict[tuple[str, str], list[FeatureFrame]] = defaultdict(list)
    taken: dict[tuple[str, str], datetime] = {}
    for one, feature in zip(picked, features, strict=True):
        for kept in _observations(one, settings):
            named = INSTRUMENTS.get(kept.iid)
            # Skip a product no instrument here builds, and one whose id names
            # no observation its instrument can be asked for.
            read = named.observation_id if named else None
            if read and (held := read(kept.pdsid)):
                wanted[(kept.iid, held)].append(feature.frame)
                taken.setdefault((kept.iid, held), kept.t_start)
    if ode is not None:
        # An instrument the selection can never name is asked which of its
        # products hold each feature's ground.
        for name in settings.instruments:
            named = INSTRUMENTS.get(name)
            if not named or not named.identifiers:
                continue
            for feature in features:
                for held in named.identifiers(feature.frame, ode):
                    wanted[(name, held)].append(feature.frame)

    jobs, skipped = [], 0
    for (instrument, identifier), held in wanted.items():
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
        # The heaviest first, so a long job is never the one left running alone.
        jobs=tuple(sorted(jobs, key=lambda job: -len(job.frames))),
        features=tuple(features),
        skipped_existing=skipped,
    )


def _sampled(picked: Sequence[Selection], settings: Settings) -> list[Selection]:
    """Keep the features one build covers, drawn evenly across their classes.

    A smaller build is drawn from the same shuffle as a larger one, so raising
    the cap adds features rather than exchanging them.

    Args:
        picked: What the search left of every feature it searched.
        settings: The settled choices for the build.

    Returns:
        The selections to build, in the order the selection was written.
    """
    kept = [one for one in picked if one.feature.kept]
    if settings.features is None or settings.features >= len(kept):
        return kept
    classes: dict[str, list[int]] = defaultdict(list)
    for at, one in enumerate(kept):
        classes[one.feature.feature_class].append(at)
    draw = random.Random(settings.seed)
    for held in classes.values():
        draw.shuffle(held)
    order = sorted(classes)
    draw.shuffle(order)
    # One from each class in turn, so every class is reached before any is
    # drawn from twice and a small build spans as many as it has room for.
    rounds = zip_longest(*(classes[name] for name in order))
    taken = [at for at in chain.from_iterable(rounds) if at is not None]
    return [kept[at] for at in sorted(taken[: settings.features])]


def _observations(one: Selection, settings: Settings):
    """Keep the observations one feature contributes, spread across its window.

    Args:
        one: What the search left of that feature.
        settings: The settled choices for the build.

    Returns:
        The observations to build, oldest first.
    """
    held = [kept for kept in one.observations if kept.iid in settings.instruments]
    cap = settings.observations_per_feature
    if cap is None:
        return held
    by_instrument: dict[str, list] = defaultdict(list)
    for kept in held:
        by_instrument[kept.iid].append(kept)
    taken = []
    for named in by_instrument.values():
        # Evenly along the window rather than the front of it, so a small build
        # keeps the spread of time the window was chosen for.
        step = max(1, len(named) // cap)
        taken.extend(named[::step][:cap])
    return sorted(taken, key=lambda kept: kept.t_start)
