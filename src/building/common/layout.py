"""How one instrument's arrays are laid out, which every stage of a build reads."""

from __future__ import annotations

from dataclasses import dataclass, field

# What an axis of a value array holds. A ground axis is the one a position
# places; the others are the instrument's own and are sampled in their own unit.
GROUND = "ground"
WAVELENGTH = "wavelength"
DELAY = "delay"


@dataclass(frozen=True, slots=True)
class Layout:
    """What an instrument's arrays hold, declared once and apart from any stage.

    Attributes:
        instrument: The instrument, as ODE names it.
        dims: What each axis of its arrays is called.
        axes: What each of those axes holds, in the same order.
        measurement: The array the instrument is stored for, which is also what
            the sample calls it.
        beside: What else of the sample is stored, keyed by the name it is
            written as and read off the sample by, each with the names of its
            own axes.
    """

    instrument: str
    dims: tuple[str, ...]
    axes: tuple[str, ...]
    measurement: str
    beside: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def ground(self) -> tuple[str, ...]:
        """Return the names of the axes a position places.

        Returns:
            The ground axes, outermost first.
        """
        return tuple(
            name
            for name, holds in zip(self.dims, self.axes, strict=True)
            if holds == GROUND
        )
