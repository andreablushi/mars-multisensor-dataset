"""Writing what the ledger holds into the records of every group it reaches."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from analysis.metadata.companions.summarise import Ledger
from analysis.metadata.fetchers.companions import stem_of
from analysis.models.companion import Companion
from common.disk.files import read_jsonl, write_jsonl


def write_companions(
    files: Iterable[Path], ledger: Ledger, companion: Companion
) -> None:
    """Give every record the medians its product's table left over its group.

    Args:
        files: The groups' records to write, each rewritten whole where it changes.
        ledger: What every table was summarised as over each group.
        companion: What the companion is, which names the fields each record gains.
    """
    for path in files:
        group = path.parent.name
        records = list(read_jsonl(path))
        changed = False
        for record in records:
            summary = ledger.get(stem_of(record["pdsid"]), {}).get(group)
            if summary is None:
                continue
            fields = {field: summary.get(field) for field in companion.columns}
            if record.items() >= fields.items():
                continue
            record.update(fields)
            changed = True
        if changed:
            write_jsonl(path, records)
