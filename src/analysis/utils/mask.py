"""How a footprint's cells are packed, written once and read back."""

from __future__ import annotations

import numpy as np

DENSE = 0
SPARSE = 1

_INDEX = np.dtype("<u4")


def encode(cells: np.ndarray, total: int) -> bytes:
    """Pack the cells a footprint fills into whichever form is smaller.

    Args:
        cells: The indices of the cells the footprint fills, in ascending order.
        total: How many cells the grid holds in all.

    Returns:
        packed: The tag byte followed by the packed cells.
    """
    packed = cells.astype(_INDEX)
    if packed.nbytes < (total + 7) // 8:
        return bytes([SPARSE]) + packed.tobytes()
    filled = np.zeros(total, dtype=bool)
    filled[cells] = True
    return bytes([DENSE]) + np.packbits(filled).tobytes()


def cells_of(mask: bytes) -> np.ndarray:
    """Return the cells one packed footprint fills.

    Args:
        mask: One footprint's mask, as encode wrote it.

    Returns:
        cells: The indices of the filled cells, in ascending order.
    """
    if mask[0] == SPARSE:
        return np.frombuffer(mask, dtype=_INDEX, offset=1)
    bits = np.unpackbits(np.frombuffer(mask, dtype=np.uint8, offset=1))
    return np.flatnonzero(bits).astype(_INDEX)
