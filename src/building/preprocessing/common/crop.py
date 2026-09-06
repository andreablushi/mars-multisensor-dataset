"""Cutting one placed observation down to the ground its own feature covers."""

from __future__ import annotations

from collections.abc import Callable

from building.metadata.models.feature import FeatureFrame
from building.preprocessing.common.cut import cut, cut_position
from building.preprocessing.common.models.crop import Crop
from building.preprocessing.common.models.cut import Cut
from building.preprocessing.common.models.relative_position import RelativePosition


def crop[Sample](
    sample: Sample,
    cut_sample: Callable[[Sample, Cut], Sample],
    position: RelativePosition,
    frame: FeatureFrame,
) -> Crop[Sample] | None:
    """Return one observation cut to the box of the feature it was kept for.

    Which samples the box keeps is the same question for every instrument, and
    cutting the arrays to them is not: which axis of one is ground differs, so
    that half is the instrument's own and is handed in.

    Args:
        sample: The observation as it was read off disk.
        cut_sample: What cuts that instrument's own arrays to what a cut keeps.
        position: Where its samples sit, against that same feature.
        frame: The local frame of the feature it was kept for.

    Returns:
        The crop, or None where the observation reaches none of the feature.
    """
    held = cut(position, frame)
    if held is None:
        return None
    return Crop(
        sample=cut_sample(sample, held),
        position=cut_position(position, held),
        inside=held.inside,
    )
