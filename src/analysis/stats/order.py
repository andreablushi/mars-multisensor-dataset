"""The one order every instrument set and instrument is drawn and tabulated in."""

from __future__ import annotations

from common.config import analysis_settings


def config_rank(name: str) -> int:
    """Rank an instrument set or an instrument by where the config first lists it.

    Args:
        name: An instrument set's key, such as MRO/CTX/EDR, or an instrument, such
            as CTX.

    Returns:
        rank: Its place in the config, past every listed one when it is not listed.
    """
    sets = analysis_settings().instrument_sets
    ranks: dict[str, int] = {}
    for rank, chosen in enumerate(sets):
        ranks.setdefault(chosen.key, rank)
        ranks.setdefault(chosen.iid, rank)
    return ranks.get(name, len(sets))
