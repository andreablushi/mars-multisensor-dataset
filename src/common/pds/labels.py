"""Reading a PDS or ISIS label, the `KEY = VALUE` text describing the file beside it."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

# The order a TRDR writes its bands in, against a DDR's band sequential.
BIL = "LINE_INTERLEAVED"

# What a label says about its own file, which stored arrays no longer need.
FILE_KEYS = frozenset(
    {
        "BANDS",
        "BAND_STORAGE_TYPE",
        "BIT_MASK",
        "BYTES",
        "COLUMNS",
        "COLUMN_NUMBER",
        "COMPRESSION_TYPE",
        "DATA_TYPE",
        "END_OBJECT",
        "FILE_RECORDS",
        "INTERCHANGE_FORMAT",
        "LINES",
        "LINE_SAMPLES",
        "OBJECT",
        "OFFSET",
        "PDS_VERSION_ID",
        "RECORD_BYTES",
        "RECORD_TYPE",
        "ROWS",
        "ROW_BYTES",
        "SAMPLE_BITS",
        "SCALING_FACTOR",
        "SAMPLE_TYPE",
        "START_BYTE",
    }
)

# What a label writes where the archive has no value to give.
MISSING = frozenset({"", "NULL", "N/A", "UNK", "UNKNOWN"})

# What a PDS sample type and width mean as a numpy dtype.
_DTYPES = {
    ("PC_REAL", 32): "<f4",
    ("PC_REAL", 64): "<f8",
    ("MSB_INTEGER", 16): ">i2",
    ("MSB_UNSIGNED_INTEGER", 16): ">u2",
    ("MSB_UNSIGNED_INTEGER", 8): "u1",
    ("UNSIGNED_INTEGER", 8): "u1",
}


def _bare_value(text: str) -> str:
    """Return a label value without its `<UNIT>` suffix and its quotes.

    Args:
        text: What the label writes after the equals sign.

    Returns:
        value: The value alone.
    """
    held = text.strip()
    # The unit comes off first, since a quoted value carries it outside its quote.
    if held.endswith(">") and "<" in held:
        held = held[: held.rindex("<")].strip()
    return held.strip('"')


def _entries(path: Path) -> Iterator[tuple[str, str]]:
    """Yield every `KEY = VALUE` line of a label, skipping comments and `{...}` lists.

    Args:
        path: The `.lbl` or `.hdr` file to read.

    Yields:
        key: The key, stripped.
        value: Its value, without unit or quotes.
    """
    in_list = False
    for line in path.read_text(errors="replace").splitlines():
        if in_list:
            in_list = "}" not in line
            continue
        # A comment is a comment, however much it looks like a key.
        if "=" not in line or line.lstrip().startswith("/*"):
            continue
        key, _, value = (part.strip() for part in line.partition("="))
        if value.startswith("{") and "}" not in value:
            in_list = True
        elif key:
            yield key, _bare_value(value)


def load(path: Path) -> dict[str, str]:
    """Read a label into a map of its keys, the first of a repeated key winning.

    Args:
        path: The `.lbl` or `.hdr` file to read.

    Returns:
        label: Each key and its value, without unit or quotes.
    """
    label: dict[str, str] = {}
    for key, value in _entries(path):
        label.setdefault(key, value)
    return label


def image_layout(label: dict[str, str]) -> tuple[int, int, int, str, str]:
    """Read the shape, band order and sample dtype of the image a label describes.

    Args:
        label: The parsed label.

    Returns:
        lines: How many lines it holds.
        samples: How many samples each line holds.
        bands: How many bands it holds.
        order: The order its bands are written in.
        dtype: The numpy dtype its samples are stored as.

    Raises:
        KeyError: When it names a sample type this cannot read.
    """
    # How many rows the image holds.
    lines = int(label["LINES"])
    # How many columns each row holds.
    samples = int(label["LINE_SAMPLES"])
    # How many channels each pixel holds, which a single band image omits.
    bands = int(label.get("BANDS", 1))
    # The order bands are written in, BIL for a TRDR and BSQ for a DDR.
    order = label.get("BAND_STORAGE_TYPE", BIL)
    # The sample type and its width, which together name a numpy dtype.
    dtype = _DTYPES[label["SAMPLE_TYPE"], int(label["SAMPLE_BITS"])]
    return lines, samples, bands, order, dtype


def columns(path: Path) -> list[dict[str, str]]:
    """Read each `OBJECT = COLUMN` block of a table label into a map of its keys.

    Args:
        path: The `.lbl` file describing the table.

    Returns:
        columns: One map per column, in the order written.
    """
    found: list[dict[str, str]] = []
    column: dict[str, str] | None = None
    for key, value in _entries(path):
        if (key, value) == ("OBJECT", "COLUMN"):
            column = {}
        elif (key, value) == ("END_OBJECT", "COLUMN") and column is not None:
            found.append(column)
            column = None
        elif column is not None:
            column[key] = value
    return found


def merge(*held: dict[str, str]) -> dict[str, str]:
    """Merge the labels of one observation's products, dropping file and unset keys.

    Args:
        held: The label of each product, the preferred first.

    Returns:
        label: The first value of every key that describes the observation itself.
    """
    merged: dict[str, str] = {}
    for one in held:
        for key, value in one.items():
            if key.startswith("^") or key in FILE_KEYS or value.upper() in MISSING:
                continue
            merged.setdefault(key, value)
    return merged
