"""Gating a SHARAD crop at its noise floor, after Carrer and Bruzzone (2017)."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from building.preprocessing.sharad.models.gate import Gate
from building.preprocessing.sharad.models.noise import Noise


def measured_noise(aligned: np.ndarray, gate: Gate) -> Noise | None:
    """Return where a crop holds only noise, and what that noise measures.

    Args:
        aligned: One crop in dB over its surface, delay samples by measured traces.
        gate: The settings its noise is measured with.

    Returns:
        noise: The crop averaged along each delay row, where its deep noise starts,
            its noise mask, and the noise's mean, spread and threshold, or None
            where no row holds an echo or no noise lies beside them.
    """
    # Average every delay row over neighbouring traces, in linear power
    power = ndimage.uniform_filter1d(
        10 ** (aligned / 10), gate.looks, axis=1, mode="nearest"
    )
    averaged = 10 * np.log10(power)
    rows = np.arange(len(averaged))[:, None]
    # Take the rows well before each trace's strongest echo as noise above the surface
    above = rows < averaged.argmax(axis=0) - gate.guard_rows
    # Flag the rows whose mean power stands over the noise above as holding an echo
    echo = power.mean(axis=1) > power[above].mean() * 10 ** (gate.echo_margin_db / 10)
    # Take the rows past the last echo as deep noise
    deep = int(np.flatnonzero(echo).max()) + gate.guard_rows + 1 if echo.any() else 0
    # Join both noise regions, leaving out every row that holds an echo
    mask = (above | (rows >= deep)) & ~echo[:, None]
    # A crop without an echo, or without noise beside its echoes, holds nothing to gate
    if not echo.any() or not mask.any():
        return None
    # Read the threshold off the noise at the false-alarm rate
    noise = averaged[mask]
    return Noise(
        averaged=averaged,
        deep=deep,
        mask=mask,
        mean=float(noise.mean()),
        spread=float(noise.std()),
        threshold=float(np.quantile(noise, 1 - gate.false_alarm)),
    )


def denoised(averaged: np.ndarray, threshold: float, spread: float) -> np.ndarray:
    """Return averaged values soft gated at a threshold, on the scale they came in."""
    # A softplus the noise spread wide fades every value under the threshold into it
    return threshold + spread * np.logaddexp(0.0, (averaged - threshold) / spread)
