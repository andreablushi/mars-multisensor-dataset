"""The selection, updated with the tiles a build dropped for an empty crop."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import replace

from analysis.selector.artifacts import read_selection, write_selection


def exclude_tiles(dropped: Collection[str]) -> None:
    """Rewrite the selection with every dropped tile no longer kept.

    Args:
        dropped: The names of the tiles a build dropped.
    """
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
            for one in read_selection()
        ]
    )
