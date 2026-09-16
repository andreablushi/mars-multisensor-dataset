"""Reading the written selection back, which every stat is read off."""

from __future__ import annotations

from analysis import dataset_list
from analysis.selector.models.selection import Selection

_picked: list[Selection] | None = None
_by_tile: dict[str, Selection] | None = None


def read_selection() -> list[Selection]:
    """Read what the selection left of every tile it searched, once.

    Returns:
        selections: One entry per tile searched, in the order they were written.

    Raises:
        FileNotFoundError: When no selection has been written.
    """
    global _picked
    if _picked is None:
        _picked = dataset_list.read_dataset_list()
    return _picked


def selection_by_tile() -> dict[str, Selection]:
    """Read the same selection keyed by the tile each row belongs to.

    Returns:
        selections: What the selection left of each tile, by tile name.

    Raises:
        FileNotFoundError: When no selection has been written.
    """
    global _by_tile
    if _by_tile is None:
        _by_tile = {one.tile.tile: one for one in read_selection()}
    return _by_tile
