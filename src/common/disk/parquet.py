"""Writing the parquet artifacts, under a schema derived from the rows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from datetime import datetime
from pathlib import Path
from types import NoneType, UnionType
from typing import Any, get_args, get_origin, get_type_hints

import pyarrow as pa
import pyarrow.parquet as pq

from common.disk.files import atomic_path

_ARROW = {
    str: pa.string(),
    int: pa.int64(),
    float: pa.float64(),
    bool: pa.bool_(),
    bytes: pa.binary(),
    datetime: pa.timestamp("us", tz="UTC"),
}


def set_type(kind: Any) -> Any:
    """Return the type an optional hint holds when set, or the hint itself."""
    # A value a row may leave unset is written as the type it holds when set
    if isinstance(kind, UnionType):
        return next(one for one in get_args(kind) if one is not NoneType)
    return kind


def schema_of(model: type) -> pa.Schema:
    """Derive the parquet schema a row model is written under.

    Args:
        model: The dataclass whose fields become the columns, in their own order.

    Returns:
        schema: The schema, every column nullable as parquet writes them.
    """
    hints = get_type_hints(model)
    columns = []
    for field in fields(model):
        kind = set_type(hints[field.name])
        # A nested row is written as its own columns, so the file gains no level
        if is_dataclass(kind):
            columns.extend(schema_of(kind))
        # A column holding many of one type is written as a list of it
        elif get_origin(kind) is tuple:
            held = set_type(get_args(kind)[0])
            columns.append((field.name, pa.list_(_ARROW[held])))
        else:
            columns.append((field.name, _ARROW[kind]))
    return pa.schema(columns)


def build[Row](model: type[Row], row: Mapping[str, Any]) -> Row:
    """Return one row model built back from the flat columns it was written as.

    Args:
        model: The dataclass to build.
        row: The columns of one written row, keyed as the schema names them.

    Returns:
        model: The model, the rows it composes built from the same flat columns.
    """
    hints = get_type_hints(model)
    held: dict[str, Any] = {}
    for field in fields(model):
        kind = set_type(hints[field.name])
        if is_dataclass(kind):
            held[field.name] = build(kind, row)
        elif get_origin(kind) is tuple:
            # A field the model holds as a tuple is written as a list, unset as a null.
            written = row[field.name]
            held[field.name] = None if written is None else tuple(written)
        else:
            held[field.name] = row[field.name]
    return model(**held)


def read_rows[Row](
    model: type[Row], schema: pa.Schema, path: Path, filters: Any = None
) -> list[Row]:
    """Read back every row of one parquet file, as the model it was written from."""
    return [
        build(model, row)
        for row in pq.read_table(path, schema=schema, filters=filters).to_pylist()
    ]


def write(data: Sequence[Any], schema: pa.Schema, path: Path) -> None:
    """Write dataclass rows to a parquet file atomically.

    Args:
        data: The dataclass rows to write.
        schema: The schema to write them under.
        path: The destination parquet file.
    """
    columns = {name: [] for name in schema.names}
    for row in data:
        for field in fields(row):
            held = getattr(row, field.name)
            # A nested row is read as its own fields are
            if is_dataclass(held):
                for one in fields(held):
                    columns[one.name].append(getattr(held, one.name))
            else:
                columns[field.name].append(held)
    table = pa.Table.from_pydict(columns, schema=schema)
    with atomic_path(path) as tmp:
        pq.write_table(table, tmp, compression="zstd")
