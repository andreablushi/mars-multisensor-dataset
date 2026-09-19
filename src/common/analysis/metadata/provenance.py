"""What a stored record says about where it came from, written and read back."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from common.analysis.models.instrument import InstrumentSet
from common.analysis.models.tile_group import TileGroup


def stamp(group: TileGroup, instrument_set: InstrumentSet, loc: str) -> dict[str, Any]:
    """Return what every record of one download carries about its origin.

    Args:
        group: The group whose box was queried.
        instrument_set: The instrument set that was asked for.
        loc: Which products the box returned.

    Returns:
        fields: The provenance fields to merge into every stored record.
    """
    return {
        "tile_group": group.name,
        "instrument_set": instrument_set.key,
        "loc_mode": loc,
        "retrieved_at": datetime.now(UTC).isoformat(),
    }


def set_key_of(item: dict[str, Any]) -> str:
    """Return the instrument set a record was downloaded for.

    Args:
        item: One stored observation record.

    Returns:
        key: The set identifier stamped on the record when it was downloaded.
    """
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
