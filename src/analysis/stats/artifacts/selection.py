"""Reading the written selection back, which every stat is read off."""

from __future__ import annotations

from functools import cache

from analysis.selector.models.selection import Selection
from analysis.utils import dataset_list


@cache
def read_selection() -> list[Selection]:
    """Read what the selection left of every tile it searched, once.

    Returns:
        selections: One entry per tile searched, in the order they were written.

    Raises:
        FileNotFoundError: When no selection has been written.
    """
    return dataset_list.read_dataset_list()


@cache
def selection_by_tile() -> dict[str, Selection]:
    """Read the same selection keyed by the tile each row belongs to.

    Returns:
        selections: What the selection left of each tile, by tile name.

    Raises:
        FileNotFoundError: When no selection has been written.
    """
    return {one.tile.tile: one for one in read_selection()}
