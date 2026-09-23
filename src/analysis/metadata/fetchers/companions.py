"""Where the archive serves the companion table of every product of a set."""

from __future__ import annotations

from pathlib import Path

from analysis.metadata.fetchers.products import every_product
from analysis.metadata.ode import ODEClient
from analysis.models.instrument import InstrumentSet
from building.download.archive import published
from common.fetch import ode

SUFFIXES = (".tab", ".lbl")


def stem_of(pdsid: str) -> str:
    """Return what a product shares with its companions, its id up to its kind.

    Args:
        pdsid: The product identifier, such as "s_00219601_rgram".

    Returns:
        stem: The identifier less its last part, such as "s_00219601".
    """
    return pdsid.rsplit("_", 1)[0]


def companion_urls(
    client: ODEClient, instrument_set: InstrumentSet, pt: str
) -> dict[str, dict[str, str]]:
    """Read where every companion table of one set is served, in one query.

    Args:
        client: The ODE client to query with.
        instrument_set: The set the companion is published beside.
        pt: The product type the companion is published under.

    Returns:
        urls: The URL of the table and of its label, by suffix, by product stem.
    """
    params = {
        "query": "product",
        "target": ode.ODE_TARGET,
        "ihid": instrument_set.ihid,
        "iid": instrument_set.iid,
        "pt": pt,
    }
    urls: dict[str, dict[str, str]] = {}
    for item in every_product(client, params, "pf"):
        for name, url in published(item).items():
            if (suffix := Path(name).suffix) in SUFFIXES:
                urls.setdefault(stem_of(item["pdsid"]), {})[suffix] = url
    return urls
