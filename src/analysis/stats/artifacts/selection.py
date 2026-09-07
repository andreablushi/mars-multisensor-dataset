"""Reading the written selection back, which every stat is read off."""

from __future__ import annotations

from analysis import dataset_list
from analysis.selector.models.selection import Selection

_picked: list[Selection] | None = None
_by_feature: dict[tuple[str, str], Selection] | None = None


def read_selection() -> list[Selection]:
    """Read what the selection left of every feature it searched, once.

    Returns:
        selections: One entry per feature searched, in the order they were written.

    Raises:
        FileNotFoundError: When no selection has been written.
    """
    global _picked
    if _picked is None:
        _picked = dataset_list.read_dataset_list()
    return _picked


def selection_by_feature() -> dict[tuple[str, str], Selection]:
    """Read the same selection keyed by the feature each row belongs to.

    Returns:
        selections: What the selection left of each feature, by class and name.

    Raises:
        FileNotFoundError: When no selection has been written.
    """
    global _by_feature
    if _by_feature is None:
        _by_feature = {
            (one.feature.feature_class, one.feature.feature_name): one
            for one in read_selection()
        }
    return _by_feature
