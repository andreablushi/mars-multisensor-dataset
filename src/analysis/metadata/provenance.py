"""What a stored record says about where it came from, written and read back."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from analysis.models.feature import Feature
from analysis.models.instrument import InstrumentSet


def stamp(feature: Feature, instrument_set: InstrumentSet, loc: str) -> dict[str, Any]:
    """Return what every record of one download carries about its origin.

    Args:
        feature: The feature whose box was queried.
        instrument_set: The instrument set that was asked for.
        loc: Which products the box returned.

    Returns:
        fields: The provenance fields to merge into every stored record.
    """
    return {
        "feature_name": feature.name,
        "feature_class": feature.feature_class,
        "feature_min_lat": feature.min_lat,
        "feature_max_lat": feature.max_lat,
        "feature_west_lon": feature.west_lon,
        "feature_east_lon": feature.east_lon,
        "instrument_set": instrument_set.key,
        "loc_mode": loc,
        "retrieved_at": datetime.now(UTC).isoformat(),
    }


def feature_of(item: dict[str, Any]) -> Feature:
    """Rebuild the feature box a record was downloaded for.

    Args:
        item: One stored observation record.

    Returns:
        feature: The feature box the record was downloaded for.
    """
    return Feature(
        name=item["feature_name"],
        feature_class=item["feature_class"],
        min_lat=float(item["feature_min_lat"]),
        max_lat=float(item["feature_max_lat"]),
        west_lon=float(item["feature_west_lon"]),
        east_lon=float(item["feature_east_lon"]),
    )


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
