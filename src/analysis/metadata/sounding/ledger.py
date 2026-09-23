"""Every track whose geometry was read, so a stopped run resumes where it left off."""

from __future__ import annotations

import json
from pathlib import Path
from typing import IO

from analysis import paths
from analysis.metadata.sounding.measure import Measured
from common.disk.files import read_jsonl

# What each track was measured as over each group, by track and then by group
Ledger = dict[str, dict[str, Measured]]


def read_ledger(path: Path = paths.SOUNDINGS_PATH) -> Ledger:
    """Read every track measured so far, a later line adding to an earlier one.

    Args:
        path: The ledger file, which need not exist.

    Returns:
        ledger: What each track was measured as over each group.
    """
    ledger: Ledger = {}
    if path.exists():
        for line in read_jsonl(path):
            ledger.setdefault(line["track"], {}).update(line["groups"])
    return ledger


def append_ledger(handle: IO[str], track: str, measured: dict[str, Measured]) -> None:
    """Write one track down the moment it is measured.

    Args:
        handle: The ledger, open for appending.
        track: The track's identifier, such as "s_00219601".
        measured: What it was measured as over each group.
    """
    handle.write(json.dumps({"track": track, "groups": measured}) + "\n")
    handle.flush()
