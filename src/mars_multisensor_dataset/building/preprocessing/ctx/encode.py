"""Encoding one CTX crop as the 8-bit WebP blocks it is stored as."""

from __future__ import annotations

from dataclasses import replace

from mars_multisensor_dataset.building.configs import ctx as configs
from mars_multisensor_dataset.building.preprocessing.ctx.models.observation import (
    ScanStatistics,
)
from mars_multisensor_dataset.building.preprocessing.ctx.models.sample import CtxSample
from mars_multisensor_dataset.common.disk.blocks import encoded_image


def encoded_sample(sample: CtxSample, statistics: ScanStatistics) -> CtxSample:
    """Return one crop with the WebP blocks it is stored as.

    Args:
        sample: The crop, its image standardised by its whole scan.
        statistics: What the whole scan is clipped and standardised by.

    Returns:
        encoded: The crop with its encoded arrays and what reads them back.
    """
    low, high = (
        (bound - statistics.mean) / statistics.std
        for bound in (statistics.low, statistics.high)
    )
    encoded, encoding = encoded_image(
        sample.image,
        low,
        high,
        configs.WEBP_BLOCK,
        configs.WEBP_QUALITY,
        configs.WEBP_METHOD,
    )
    return replace(sample, encoded=encoded, encoding=encoding)
