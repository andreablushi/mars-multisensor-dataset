"""Reading the fixed width ASCII table a PDS label describes."""

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
    """Parse a PDS timestamp as UTC, a rounded up `:60.` second read as the next minute.

    Args:
        text: The timestamp as the archive wrote it.

    Returns:
        moment: The timestamp, timezone aware.

    Raises:
        ValueError: When the text is not a timestamp at all.
    """
    text = text.strip()
    head, rolled, rest = text.rpartition(ROLLED)
    if rolled:
        moment = datetime.fromisoformat(f"{head}:00.{rest}") + timedelta(minutes=1)
    else:
        moment = datetime.fromisoformat(text)
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


def _time_column(text: np.ndarray) -> np.ndarray:
    """Parse a TIME column, rewriting each rounded up `:60.` second first.

    Args:
        text: The column's values, one fixed width byte string per row.

    Returns:
        times: The times, as naive millisecond datetimes.

    Raises:
        ValueError: When a stamp is not one that can be read at all.
    """
    values = text.astype("U")
    for at in np.flatnonzero(np.char.find(values, ROLLED) > 0):
        moment = parse_timestamp(values[at]).replace(tzinfo=None)
        values[at] = moment.isoformat(timespec="milliseconds")
    return values.astype(_DTYPES["TIME"])


def build_table(
    table: Path, label: dict[str, str], fields: list[dict[str, str]]
) -> np.recarray:
    """Read a `.tab` table's ROWS records, slicing each COLUMN at its START_BYTE.

    Args:
        table: The `.tab` file holding the records.
        label: The parsed label giving ROWS and ROW_BYTES.
        fields: The COLUMN objects to read, as `labels.columns` returns them.

    Returns:
        table: One row per record, each column named and typed as its label says.
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
        if field["DATA_TYPE"] == "TIME":
            built[field["NAME"]] = _time_column(text)
        else:
            built[field["NAME"]] = text.astype(_DTYPES.get(field["DATA_TYPE"], "f8"))
    return np.rec.fromarrays(list(built.values()), names=list(built))


def load_table(table: Path) -> tuple[np.recarray, dict[str, str]]:
    """Read a whole `.tab` table and the `.lbl` label beside it.

    Args:
        table: The `.tab` file, whose `.lbl` sits beside it.

    Returns:
        table: One row per record, each column named as the label names it.
        label: The parsed label.

    Raises:
        FileNotFoundError: When the table or its label is missing.
    """
    path = table.with_suffix(".lbl")
    label = labels.load(path)
    return build_table(table, label, labels.columns(path)), label
