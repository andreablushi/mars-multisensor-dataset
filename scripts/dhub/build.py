"""Building one dataset on DigitalHub, published as it goes."""

from __future__ import annotations

from collections.abc import Sequence

from analysis.selector.models.selection import Selection
from common.building import build, paths
from common.building.models.settings import Settings
from dhub import archives
from dhub import configs as platform

# What the published dataset holds, said once since a checkpoint publishes it too.
DATASET_HELD = (
    "The cropped observations and their index, one object per crop; read "
    "observations.parquet and ask the store for the crops it names."
)


def published_dataset(
    project,
    settings: Settings,
    picked: Sequence[Selection],
    force: bool = False,
):
    """Build one dataset on DigitalHub and publish what it left on disk.

    Args:
        project: The DigitalHub project the dataset is logged into.
        settings: The settled choices for the build, whose name the dataset is
            published under so one never overwrites another.
        picked: The tiles to build, each with the observations its window keeps.
        force: Whether to build the dataset again from nothing, rather than
            filling in whatever the last build of it left missing.

    Returns:
        dataset: The published dataset, one object per crop.

    Raises:
        RuntimeError: When a product failed, which leaves the dataset short of
            what the selection asked for.
    """
    published = platform.load().publishes
    name = settings.name
    published_as = f"{published['dataset']}-{name}"
    root = paths.dataset_root(name)
    # A job starts on an empty disk, so only the index of what is built comes down
    if not force:
        archives.download_files(project, published_as, root, paths.INDEX_NAMES)
    print(f"building the dataset as {name}", flush=True)

    def checkpoint():
        """Publish the dataset as it stands, then delete the crops it sent from disk.

        Returns:
            dataset: The published dataset.
        """
        crops = paths.crop_paths(root)
        # Whatever sits beside the crops describes them, so it goes up after them
        index = sorted(one for one in root.iterdir() if one.is_file())
        dataset = archives.published_folder(
            project, root, crops, index, published_as, DATASET_HELD, settings.workers
        )
        for crop in crops:
            crop.unlink()
        return dataset

    failed = build.build_dataset(settings, picked, force, checkpoint)
    dataset = checkpoint()
    if failed:
        raise RuntimeError(
            "the build had failures; what was published holds what finished"
        )
    print("done", flush=True)
    return dataset
