"""How the tiles are labelled, once its config has been read."""

from __future__ import annotations

from dataclasses import dataclass

from mars_multisensor_dataset.analysis.ground_truth.swim.models import SwimSettings

from ..nomenclature.models.settings import (
    NomenclatureSettings,
)


@dataclass(slots=True)
class GroundTruthSettings:
    """The settled choices of every source the tiles are labelled from.

    Attributes:
        nomenclature: How the evaluation classes are read from the IAU nomenclature.
        swim: How the SWIM ice maps are sampled under every tile.
    """

    nomenclature: NomenclatureSettings
    swim: SwimSettings
