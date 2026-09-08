"""Which coverage constraints a window meets, and how much ground answers each."""

from __future__ import annotations

from collections.abc import Sequence

from analysis.selector.models.filter import Constraints


def coverage_constraints(
    constraints: Constraints, cells_reached: Sequence[int]
) -> list[int] | None:
    """Take what each constraint reaches, or refuse them all when one goes unmet.

    Args:
        constraints: The sets answering each one and their floors, tightest first.
        cells_reached: How many cells each set reaches inside the window.

    Returns:
        counts: The cells each constraint reaches, in order, or None when one is unmet.
    """
    counts: list[int] = []
    for answers in constraints:
        # A constraint is answered by whichever instrument reaches most of its bar
        cell_count = 0
        for answering, floor in answers:
            reached = (
                cells_reached[answering[0]]
                if len(answering) == 1
                else max((cells_reached[owner] for owner in answering), default=0)
            )
            if reached < floor:
                continue
            if reached > cell_count:
                cell_count = reached
        if not cell_count:
            return None
        counts.append(cell_count)
    return counts
