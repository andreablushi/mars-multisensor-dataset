"""One field a timeless instrument's looks are ranked by, the first rule deciding."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(slots=True)
class Rank:
    """One companion field a look is ranked by.

    Attributes:
        iid: The instrument whose looks it ranks.
        name: The field it reads, as the companion writes it into the records.
        above: The bar a look past which ranks first, or None to rank the lowest first.
    """

    iid: str
    name: str
    above: float | None = None

    def placed(self, values: Mapping[str, float]) -> float:
        """Return where one look stands under this rule, the best lowest.

        Args:
            values: The fields the look's record carries, by name.

        Returns:
            place: Nought past the bar and one short of it, or the value itself.
        """
        value = values.get(self.name)
        if value is None:
            return math.inf
        return value if self.above is None else float(value <= self.above)
