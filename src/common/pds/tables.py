"""Reading the ASCII table a PDS label describes, column by column."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from common.pds import labels

# What each column type is read as, floats where the label names nothing else.
_DTYPES = {"ASCII_INTEGER": "i8", "TIME": "M8[ms]"}

# How an archive writes a second it rounded up, which no calendar holds.
ROLLED = ":60."


def parse_timestamp(text: str) -> datetime:
    """Return one archive timestamp in UTC, a rounded up second read as the next minute.

    Args:
        text: The timestamp as the archive wrote it.

    Returns:
        moment: The timestamp, timezone aware.

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


def _times(text: np.ndarray) -> np.ndarray:
    """Read a time column, rolling a rounded up second into the minute after it.

    Args:
        text: The column's values, one fixed width byte string per row.

    Returns:
        times: The times, as datetimes.

    Raises:
        ValueError: When a stamp is not one that can be read at all.
    """
    values = text.astype("U")
    rolled = np.flatnonzero(np.char.find(values, ROLLED) > 0)
    if rolled.size:
        stamps = values.tolist()
        for at in rolled:
            whole = parse_timestamp(stamps[at]).replace(tzinfo=None)
            stamps[at] = whole.isoformat(timespec="milliseconds")
        values = np.array(stamps, dtype=values.dtype)
    return values.astype(_DTYPES["TIME"])


def build_table(table: Path, label: dict[str, str], fields: list[dict[str, str]]):
    """Read one fixed width ASCII table into a row per record.

    Args:
        table: The `.tab` file holding the records.
        label: The parsed label describing it.
        fields: The COLUMN objects, as `labels.columns` returns them.

    Returns:
        table: One row per record, its fields named and typed as the label says.
    """
    rows, width = int(label["ROWS"]), int(label["ROW_BYTES"])
    raw = np.fromfile(table, dtype="S1", count=rows * width)
    # One fixed width record per row, so a column is a slice of every one.
    records = raw.reshape(rows, width)
    built = {}
    for field in fields:
        # START_BYTE is written one based, and BYTES is the width that follows.
        start = int(field["START_BYTE"]) - 1
        cut = records[:, start : start + int(field["BYTES"])]
        text = np.char.strip(cut.view(f"S{cut.shape[1]}").reshape(rows))
        built[field["NAME"]] = (
            _times(text)
            if field["DATA_TYPE"] == "TIME"
            else text.astype(_DTYPES.get(field["DATA_TYPE"], "f8"))
        )
    return np.rec.fromarrays(list(built.values()), names=list(built))


def load_table(table: Path) -> tuple[np.recarray, dict[str, str]]:
    """Read one table and the label beside it that describes it.

    Args:
        table: The `.tab` file holding the records, whose `.lbl` sits beside it.

    Returns:
        table: One row per record, its fields named as the label names its columns.
        label: The parsed label describing them.

    Raises:
        FileNotFoundError: When the table or its label is missing.
    """
    path = table.with_suffix(".lbl")
    label = labels.load(path)
    return build_table(table, label, labels.columns(path)), label
