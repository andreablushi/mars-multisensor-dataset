"""Running one dataset build where it was asked for, here or on DigitalHub."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from analysis.selector.models.selection import Selection
from building.models.settings import BuildSettings
from building.runner import build_dataset
from dhub import args, submit
from dhub.paths import Function


def build_exit_code[T: BuildSettings](
    description: str,
    stage: Function,
    settled: Callable[[int | None, Sequence[str]], T],
    selections: Callable[[T], list[Selection]],
) -> int:
    """Parse a build script's flags, then submit the build or run it here.

    Args:
        description: What the script does, shown by --help.
        stage: The stage to submit when --dh is given.
        settled: What settles the build, given its cores and the overrides it was
            started with.
        selections: What reads the tiles to build once the build is settled.

    Returns:
        code: A process exit code, non zero when a product failed.
    """
    parser = args.script_parser(description)
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, as key=value")
    arguments = parser.parse_args()

    if arguments.dh:
        return submit.submitted(
            stage, arguments.ref, arguments.overrides, force=arguments.force
        )
    settings = settled(None, arguments.overrides)
    return build_dataset(settings, selections(settings), arguments.force)
