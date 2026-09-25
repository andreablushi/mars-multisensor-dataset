"""How one instrument's arrays are laid out, which every stage of a build reads."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Axis(StrEnum):
    """What an axis holds: ground is placed, the others are the instrument's own."""

    GROUND = "ground"
    WAVELENGTH = "wavelength"
    DELAY = "delay"


@dataclass(frozen=True, slots=True)
class Layout:
    """What an instrument's arrays hold, declared once and apart from any stage.

    Attributes:
        instrument: The instrument, as ODE names it.
        dims: The name of each axis of its measurement.
        axes: What each of those axes holds, in the same order.
        measurement: The sample attribute holding the main array, and its stored name.
        beside: The other arrays stored with it, by name, with the dims each spans.
        stored: The dtype the measurement is written as, or None to keep its own.
        band_centres_nm: The band centres every sample shares, in nm, or None when
            the instrument has no wavelength axis.
    """

    instrument: str
    dims: tuple[str, ...]
    axes: tuple[Axis, ...]
    measurement: str
    beside: dict[str, tuple[str, ...]] = field(default_factory=dict)
    stored: str | None = None
    band_centres_nm: tuple[float, ...] | None = None

    def axis_indices(self, kind: Axis) -> tuple[int, ...]:
        """Return where every axis holding one kind sits in the measurement."""
        return tuple(at for at, holds in enumerate(self.axes) if holds == kind)
