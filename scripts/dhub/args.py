"""What every script is started with, and how it stops when interrupted."""

from __future__ import annotations

import argparse
from collections.abc import Callable

from common.console import print_interrupted

REBUILT = (
    "build the dataset again from nothing, rather than filling in what the last "
    "build left missing"
)


def script_parser(description: str, forced: str = REBUILT) -> argparse.ArgumentParser:
    """Return the parser every script shares, for it to add its own flags to.

    Args:
        description: What the script does, shown by --help.
        forced: What --force redoes in this script.

    Returns:
        parser: The parser holding --dh, --force and --ref.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--dh", action="store_true", help="submit to DigitalHub instead of running here"
    )
    parser.add_argument("--force", action="store_true", help=forced)
    parser.add_argument("--ref", default="main", help="branch, tag, or commit to run")
    return parser


def run_script(main: Callable[[], int], kept: str) -> None:
    """Run a script's entry point and exit with its code, even when interrupted.

    Args:
        main: The entry point, handing back a process exit code.
        kept: What an interrupted run leaves behind on disk.
    """
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print_interrupted(kept)
        raise SystemExit(130) from None
