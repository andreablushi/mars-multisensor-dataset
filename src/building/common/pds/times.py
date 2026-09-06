"""Reading a timestamp an archive wrote, however it rounded the second."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

# How an archive writes a second it rounded up, which no calendar holds.
ROLLED = ":60."


def moment(text: str) -> datetime:
    """Return one archive timestamp, in UTC.

    Args:
        text: The timestamp as the archive wrote it.

    Returns:
        The timestamp, a rounded up second read as the minute after it.

    Raises:
        ValueError: When the text is not a timestamp at all.
    """
    head, rolled, rest = text.strip().rpartition(ROLLED)
    held = (
        datetime.fromisoformat(f"{head}:00.{rest}") + timedelta(minutes=1)
        if rolled
        else datetime.fromisoformat(text.strip())
    )
    return held if held.tzinfo else held.replace(tzinfo=UTC)
