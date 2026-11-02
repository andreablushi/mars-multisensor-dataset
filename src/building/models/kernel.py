"""One kernel an ISIS kernel database offers, and when it covers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True, slots=True)
class KernelSelection:
    """One reconstructed kernel a database selects, and the spans it covers.

    Attributes:
        path: Where the kernel belongs under ISISDATA.
        url: Where NAIF serves it.
        spans: The first and last moment of each span it covers, in TDB.
    """

    path: Path
    url: str
    spans: tuple[tuple[datetime, datetime], ...]

    def covers(self, start: datetime, stop: datetime) -> bool:
        """Return whether any of its spans overlaps a stretch of time."""
        return any(first <= stop and start <= last for first, last in self.spans)
