"""Where the archive serves every SHARAD track's geometry table."""

from __future__ import annotations

from analysis.metadata.fetchers.products import every_product
from analysis.metadata.ode import ODEClient
from analysis.models.instrument import InstrumentSet
from common.fetch import ode

# The ODE product type every track's geometry is published under
GEOMETRY_PT = "USGEOMV2"

TABLE_SUFFIX = ".tab"


def track_of(pdsid: str) -> str:
    """Return the track one of its products belongs to.

    Args:
        pdsid: The product identifier, such as "s_00219601_rgram".

    Returns:
        track: The track, such as "s_00219601".
    """
    return pdsid.rsplit("_", 1)[0]


def geometry_urls(client: ODEClient, sounder: InstrumentSet) -> dict[str, str]:
    """Read where every geometry table of one sounder is served, in one query.

    Args:
        client: The ODE client to query with.
        sounder: The radargram set the geometry is published beside.

    Returns:
        urls: The table's URL, by the track it belongs to.
    """
    params = {
        "query": "product",
        "target": ode.ODE_TARGET,
        "ihid": sounder.ihid,
        "iid": sounder.iid,
        "pt": GEOMETRY_PT,
    }
    urls: dict[str, str] = {}
    for item in every_product(client, params, "pf"):
        offered = item.get("Product_files", {}).get("Product_file", [])
        for offer in offered if isinstance(offered, list) else [offered]:
            if str(offer.get("FileName", "")).lower().endswith(TABLE_SUFFIX):
                urls[track_of(item["pdsid"])] = offer["URL"]
    return urls
