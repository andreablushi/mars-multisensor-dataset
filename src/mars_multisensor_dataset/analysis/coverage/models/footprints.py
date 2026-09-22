"""One instrument set's footprints, parsed once and indexed for every tile."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely import STRtree


@dataclass(frozen=True, slots=True)
class ParsedFootprints:
    """Every footprint of one set, in lon/lat and on each pole, ready to be queried.

    Attributes:
        geographic: The footprints in lon/lat degrees, one per observation, indexed.
        polar: The footprints ODE published in polar stereographic metres, by pole.
        swath_widths_m: The width each track is buffered to, zero for an area.
    """

    geographic: STRtree
    polar: dict[bool, STRtree]
    swath_widths_m: np.ndarray
