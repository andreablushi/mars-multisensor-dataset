"""Reading the ASCII table a PDS label describes, column by column."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from building.common.pds import labels, times

# What each column type is read as, floats where the label names nothing else.
_DTYPES = {"ASCII_INTEGER": "i8", "TIME": "M8[ms]"}


def _times(text: np.ndarray) -> np.ndarray:
    """Read a time column, rolling a rounded up second into the minute after it.

    Args:
        text: The column's values, one fixed width byte string per row.

    Returns:
        The times, as datetimes.

    Raises:
        ValueError: When a stamp is not one that can be read at all.
    """
    values = text.astype("U")
    rolled = np.flatnonzero(np.char.find(values, times.ROLLED) > 0)
    if rolled.size:
        stamps = values.tolist()
        for at in rolled:
            whole = times.moment(stamps[at]).replace(tzinfo=None)
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
        A structured array of one row per record, its fields named as the label
        names them, integers read as integers, times as datetimes and the rest
        as floats.
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
        A structured array of one row per record, its fields named as the label
        names its columns, and the parsed label describing them.

    Raises:
        FileNotFoundError: When the table or its label is missing.
    """
    path = table.with_suffix(".lbl")
    label = labels.load(path)
    return build_table(table, label, labels.columns(path)), label
