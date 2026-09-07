"""Laying the tiles one feature stands on onto the single grid they share."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from building.common.pds import images, labels
from building.models.feature import FeatureFrame
from building.preprocessing.mola import projection
from building.preprocessing.mola.models.grid import MolaGrid
from building.preprocessing.mola.models.observation import MolaObservation
from utils.geometry import geodesy

TURN = 360.0


def merge_tiles(grid: MolaGrid, frame: FeatureFrame) -> MolaObservation:
    """Return the one grid every tile a feature stands on writes its part of.

    Args:
        grid: The tiles of the grid that landed, and how fine it is.
        frame: The local frame of the feature the tiles are merged for.

    Returns:
        The observation holding that feature's own box and no more of the grid,
        its longitudes running past a whole turn where the box crosses the
        meridian.

    Raises:
        FileNotFoundError: When a tile's label is missing.
        KeyError: When a label names a sample type this cannot read.
        ValueError: When a label names a projection this cannot read, or the
            tiles that landed leave any part of the box unwritten.
    """
    resolution = grid.resolution

    def covered() -> tuple[range, range]:
        """Return which bins of the whole planet's grid the box covers.

        Returns:
            The lines it covers, counted south from the north pole, and the
            samples, counted east from the meridian and running past a whole
            turn where the box crosses it.
        """
        span = geodesy.longitude_span(frame.west_lon, frame.east_lon)
        return (
            range(
                math.ceil((90.0 - frame.max_lat) * resolution - 0.5),
                math.floor((90.0 - frame.min_lat) * resolution - 0.5) + 1,
            ),
            range(
                math.ceil(frame.west_lon * resolution - 0.5),
                math.floor((frame.west_lon + span) * resolution - 0.5) + 1,
            ),
        )

    def placed(label: dict[str, str]) -> tuple[int, int]:
        """Return where one tile's first bin sits on the grid they all share.

        Args:
            label: The parsed label of the tile.

        Returns:
            The line and the sample of the whole planet's grid that the tile's
            own first line and first sample are.
        """
        latitude, longitude = projection.load(label)
        return (
            round((90.0 - float(latitude[0])) * resolution - 0.5),
            round(float(longitude[0]) * resolution - 0.5) % (round(TURN) * resolution),
        )

    def window(
        image: Path,
        label: dict[str, str],
        lines: tuple[int, int],
        samples: tuple[int, int],
    ) -> np.ndarray:
        """Read only the rectangle of one tile the bounds ask for.

        Args:
            image: The `.img` file holding the values.
            label: The parsed label describing it.
            lines: The first line to read, and the line after the last.
            samples: The first sample to read, and the sample after the last.

        Returns:
            The values inside those bounds, as lines by samples, in the unit
            the label says they stand for.

        Raises:
            ValueError: When the tile holds more than the one band this reads.
        """
        down, across, bands, _, dtype = labels.layout(label)
        if bands != 1:
            raise ValueError(f"{image.name} holds {bands} bands rather than one.")
        with image.open("rb") as handle:
            handle.seek(lines[0] * across * np.dtype(dtype).itemsize)
            flat = np.fromfile(
                handle, dtype=dtype, count=(lines[1] - lines[0]) * across
            )
        held = flat.reshape(lines[1] - lines[0], across)[:, samples[0] : samples[1]]
        return images.measured(held, label)

    down, across = covered()
    whole = round(TURN) * resolution
    height: np.ndarray | None = None
    written = np.zeros((len(down), len(across)), dtype=bool)
    read = []
    for tile, image in sorted(grid.files.items()):
        label = labels.load(image.with_suffix(".lbl"))
        read.append(label)
        line, sample = placed(label)
        lines, samples = int(label["LINES"]), int(label["LINE_SAMPLES"])
        # A box running over the meridian meets a tile a whole turn along, too.
        for shift in (0, whole):
            first, last = max(down.start, line), min(down.stop, line + lines)
            starts = max(across.start, sample + shift)
            stops = min(across.stop, sample + shift + samples)
            if first >= last or starts >= stops:
                continue
            part = window(
                image,
                label,
                (first - line, last - line),
                (starts - sample - shift, stops - sample - shift),
            )
            if height is None:
                height = np.zeros((len(down), len(across)), dtype=part.dtype)
            at = np.s_[
                first - down.start : last - down.start,
                starts - across.start : stops - across.start,
            ]
            height[at] = part
            written[at] = True
    if height is None or not written.all():
        raise ValueError(
            f"{frame.feature_name} reaches ground no tile of {grid.name} holds."
        )
    return MolaObservation(
        grid.name,
        labels.merge(*read),
        height,
        90.0 - (np.arange(down.start, down.stop) + 0.5) / resolution,
        (np.arange(across.start, across.stop) + 0.5) / resolution,
    )
