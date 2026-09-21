"""Reading a PDS or ISIS label, and whatever it says about what sits beside it."""

from __future__ import annotations

from pathlib import Path

# The order a TRDR writes its bands in, against a DDR's band sequential.
BIL = "LINE_INTERLEAVED"

# What a label says about its own file, which stored arrays no longer need.
LAYOUT = frozenset(
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


def _value(text: str) -> str:
    """Return one label value, its unit suffix and then its quotes stripped.

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


def load(path: Path) -> dict[str, str]:
    """Read a label into its keys and values.

    Args:
        path: The `.lbl` or `.hdr` file to read.

    Returns:
        label: The label, keyed as written, quotes and unit suffixes stripped, the first
            of a repeated key winning.
    """
    label: dict[str, str] = {}
    skipping = False
    for line in path.read_text(errors="replace").splitlines():
        if skipping:
            skipping = "}" not in line
            continue
        # A comment is a comment, however much it looks like a key.
        if "=" not in line or line.lstrip().startswith("/*"):
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if value.startswith("{") and "}" not in value:
            skipping = True
            continue
        key = key.strip()
        if key and key not in label:
            label[key] = _value(value)
    return label


def layout(label: dict[str, str]) -> tuple[int, int, int, str, str]:
    """Read how one image is shaped and written from its label.

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
    stored = label.get("BAND_STORAGE_TYPE", BIL)
    # The sample type and its width, which together name a numpy dtype.
    dtype = _DTYPES[label["SAMPLE_TYPE"], int(label["SAMPLE_BITS"])]
    return lines, samples, bands, stored, dtype


def scaling(label: dict[str, str]) -> tuple[float, float]:
    """Read what one image's stored values have to be turned into to be read.

    Args:
        label: The parsed label.

    Returns:
        factor: What every stored value is multiplied by, one where the label names
            none.
        offset: What is added after it, zero where the label names none.
    """
    return float(label.get("SCALING_FACTOR", 1.0)), float(label.get("OFFSET", 0.0))


def columns(path: Path) -> list[dict[str, str]]:
    """Read the COLUMN objects one table label names, in the order written.

    Args:
        path: The `.lbl` file describing the table.

    Returns:
        columns: One dictionary per column, keyed as the label writes it, stripped of
            quotes and units.
    """
    found: list[dict[str, str]] = []
    inside: dict[str, str] | None = None
    for line in path.read_text(errors="replace").splitlines():
        key, _, value = (part.strip() for part in line.partition("="))
        if key == "OBJECT" and value == "COLUMN":
            inside = {}
        elif key == "END_OBJECT" and value == "COLUMN" and inside is not None:
            found.append(inside)
            inside = None
        elif inside is not None and key:
            inside[key] = _value(value)
    return found


def merge(*held: dict[str, str]) -> dict[str, str]:
    """Return one label for an observation published as several products.

    Args:
        held: The label of each product, in the order they are preferred.

    Returns:
        label: Their keys in one map, without what only describes the file they came in
            and without a key the archive left unset.
    """
    merged: dict[str, str] = {}
    for one in held:
        for key, value in one.items():
            kept = key not in merged and not key.startswith("^") and key not in LAYOUT
            if kept and value.upper() not in MISSING:
                merged[key] = value
    return merged
