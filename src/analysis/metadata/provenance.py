"""What a stored record says about where it came from, written and read back."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from analysis.models.instrument import InstrumentSet


def stamp(instrument_set: InstrumentSet) -> dict[str, Any]:
    """Return the provenance fields every record of one download carries."""
    return {"instrument_set": instrument_set.key}


def set_key_of(item: dict[str, Any]) -> str:
    """Return the key of the instrument set a record was downloaded for."""
    return str(item["instrument_set"])


def as_utc(text: str) -> datetime:
    """Parse an ODE timestamp as UTC.

    Args:
        text: The ISO 8601 timestamp as stored.

    Returns:
        moment: The timezone-aware timestamp.
    """
    parsed = datetime.fromisoformat(text)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
