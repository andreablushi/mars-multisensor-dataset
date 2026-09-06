"""The ground a feature covers, and which samples of an observation fall on it."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Box:
    """One feature's extent, as offsets from the centre its frame is built on.

    ODE publishes a feature as a box and never as an outline, so this is the
    whole of what is known about where a feature lies.

    Attributes:
        south: How far south of the centre the box reaches, in degrees, below
            zero where the centre is the middle of it.
        north: How far north of the centre it reaches, in degrees.
        west: The offset of its western edge, in degrees, wrapped to -180..180.
        span: How far east it runs from that edge, in degrees, above zero and
            up to 360 for a feature running through every longitude.
    """

    south: float
    north: float
    west: float
    span: float


@dataclass(frozen=True, slots=True)
class Cut:
    """What one feature's box keeps of one observation.

    Attributes:
        bounds: The samples to keep of each ground axis, in the position's
            order.
        inside: Which of the samples that survives the cut truly falls in the
            box, or None where every one of them does. A map raster meets a box
            in a rectangle and so is left unset, and only a swath crossing the
            box or a track grazing it has corners to mark.
    """

    bounds: tuple[np.ndarray, ...]
    inside: np.ndarray | None
