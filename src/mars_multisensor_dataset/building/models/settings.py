"""What one build was asked to do, once the config has been read."""

from __future__ import annotations

from dataclasses import dataclass

from omegaconf import MISSING

from mars_multisensor_dataset.building.preprocessing.sharad.models.gate import Gate


@dataclass(slots=True)
class CtxCodec:
    """How a CTX crop is stored as blocks of lossy WebP.

    Attributes:
        quality: The WebP quality, 0 to 100.
        method: The WebP encoder effort, 0 to 6.
        block: How many pixels a side one block holds.
    """

    quality: int
    method: int
    block: int


@dataclass(slots=True)
class CtxPreprocessing:
    """What the CTX reader is handed.

    Attributes:
        clip: The low and high percentiles a scan's counts are clipped at.
        stride: How many lines and samples apart a scan's statistics are measured.
        pool: How many pixels a side every crop is averaged over.
        square_ground: Whether an equatorial crop is projected with square ground
            pixels at its tile's latitude, rather than square in degrees.
        codec: How every crop is stored as WebP blocks, or None for float16.
    """

    clip: list[int]
    stride: int
    pool: int
    square_ground: bool
    codec: CtxCodec | None = None


@dataclass(slots=True)
class SpikeSettings:
    """How CRISM spikes are removed, as crism_ml sets it.

    Attributes:
        passes: How many times the cube is filtered.
        window: How many bands wide the moving median is.
        sigma: How many spreads from the median a value may sit.
    """

    passes: int
    window: int
    sigma: float


@dataclass(slots=True)
class CrismPreprocessing:
    """What the CRISM reader is handed.

    Attributes:
        spikes: How spikes are removed.
        levelling: How many lines a column is levelled over along the track.
        join_ratio: The lowest and highest ratio of the first infrared band to the
            last visible one an observation keeps.
    """

    spikes: SpikeSettings
    levelling: int
    join_ratio: list[float]


@dataclass(slots=True)
class Preprocessing:
    """What each instrument's reader is handed, the same in every build.

    Attributes:
        CTX: The CTX reader's settings.
        CRISM: The CRISM reader's settings.
        SHARAD: The noise gate every SHARAD crop is measured with.
    """

    CTX: CtxPreprocessing
    CRISM: CrismPreprocessing
    SHARAD: Gate


@dataclass(slots=True)
class BuildSettings:
    """The settled choices for a build, whichever dataset it builds.

    Attributes:
        name: The build's name, its directory and what it is published under.
        workers: How many products are built at once, one per core.
        downloads: How many downloads run at once from each archive, by its name.
        preprocessing: What each instrument's reader is handed.
        reference: The build whose constants this one is standardised by, or None to
            pool its own.
        instruments: The instruments it builds, or None for every one.
    """

    name: str
    workers: int
    downloads: dict[str, int]
    preprocessing: Preprocessing
    reference: str | None = None
    instruments: list[str] | None = None


@dataclass(slots=True)
class TrainingSettings(BuildSettings):
    """The settled choices for the training build, beside how every build runs.

    Attributes:
        share: The share of kept tiles to build, above zero up to one.
        seed: The draw's seed, so a smaller build is a reproducible subset.
    """

    share: float = MISSING
    seed: int = MISSING

    @property
    def prunes_selection(self) -> bool:
        """Return whether it refuses what it cropped empty, as a full build does."""
        return self.instruments is None
