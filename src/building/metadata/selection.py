"""The selection, updated with the tiles a build dropped for an empty crop."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from analysis.selector.artifacts import (
    read_selection,
    write_refused_observations,
    write_selection,
)
from building.models.instrument import observation_identifier
from building.models.job import Outcome


def exclude_tiles(outcomes: Sequence[Outcome]) -> None:
    """Drop every tile a build cropped empty, refusing the observations that emptied it.

    Args:
        outcomes: What every job of the build left.
    """
    emptied = {
        (name, one.job.instrument, one.job.identifier)
        for one in outcomes
        for name in one.emptied
    }
    dropped = {name for name, _, _ in emptied}
    selections = read_selection()
    write_refused_observations(
        [
            observation
            for one in selections
            for observation in one.observations
            if (
                observation.tile,
                observation.iid,
                observation_identifier(observation.iid, observation.pdsid),
            )
            in emptied
        ]
    )
    write_selection(
        [
            replace(
                one,
                tile=replace(
                    one.tile,
                    kept=False,
                    start=None,
                    end=None,
                    days=0.0,
                    geo_mean=0.0,
                    taken=0,
                ),
                observations=[],
            )
            if one.tile.tile in dropped
            else one
            for one in selections
        ]
    )
