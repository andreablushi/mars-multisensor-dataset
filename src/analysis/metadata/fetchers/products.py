"""Fetching one group and instrument set's product records, a page at a time."""

from __future__ import annotations

from typing import Any, TypeAlias

import analysis.metadata.provenance as provenance
from analysis.metadata.ode import ODEClient
from analysis.models.instrument import InstrumentSet
from shared.fetch import ode
from shared.fetch.ode import ODEError
from shared.models.tile_group import TileGroup

# A group circling a pole is asked in two halves, no ODE box reaching round
LONGITUDE_HALVES = ((0.0, 180.0), (180.0, 360.0))

PAGE_SIZE = 5000
PAGE_ORDER = "oba"

RETAINED_FIELDS = (
    "pdsid",
    "ihid",
    "iid",
    "pt",
    "Map_scale",
    "UTC_start_time",
    "UTC_stop_time",
    "Minimum_latitude",
    "Maximum_latitude",
    "Westernmost_longitude",
    "Easternmost_longitude",
    "Footprint_C0_geometry",
)

ProductRecord: TypeAlias = dict[str, Any]


def fetch_products(
    client: ODEClient,
    group: TileGroup,
    instrument_set: InstrumentSet,
    loc: str,
) -> list[ProductRecord]:
    """Fetch all product metadata for a group and instrument set.

    Args:
        client: The ODE client to query with.
        group: The group whose box the query is built from.
        instrument_set: The instrument host, instrument, and product type.
        loc: Which products the box returns, recorded with each one.

    Returns:
        products: One record per distinct product, in the order ODE returned them.

    Raises:
        ODEError: When ODE reports no usable count for a box.
    """
    stamped = provenance.stamp(group, instrument_set, loc)
    records: list[ProductRecord] = []
    # The two boxes a polar group is asked in overlap, so a product returns twice
    seen: set[tuple[str, str]] = set()
    spans = (
        LONGITUDE_HALVES
        if group.circles_a_pole
        else ((group.west_lon, group.east_lon),)
    )
    for west_lon, east_lon in spans:
        params = {
            "query": "product",
            "target": ode.ODE_TARGET,
            "ihid": instrument_set.ihid,
            "iid": instrument_set.iid,
            "pt": instrument_set.pt,
            "minlat": str(group.min_lat),
            "maxlat": str(group.max_lat),
            "westernlon": str(west_lon),
            "easternlon": str(east_lon),
            "loc": loc,
        }
        if instrument_set.product_id:
            params["productid"] = instrument_set.product_id
        raw = client.query({**params, "results": "c"}).get("Count")
        try:
            total = int(raw)
        except (TypeError, ValueError):
            raise ODEError(f"ODE returned no product count, found {raw!r}") from None
        offset = 0
        while offset < total:
            page = client.query(
                {
                    **params,
                    "results": "opm",
                    "order": PAGE_ORDER,
                    "limit": str(PAGE_SIZE),
                    "offset": str(offset),
                }
            )
            found = page["Products"]["Product"]
            # A box holding one product is answered with that product, not a list of one
            items = found if isinstance(found, list) else [found]
            # Nothing to advance by would page the same offset forever
            if not items:
                break
            for item in items:
                identity = (item["Footprint_C0_geometry"], item["UTC_start_time"])
                if identity in seen:
                    continue
                seen.add(identity)
                kept = {f: item[f] for f in RETAINED_FIELDS if f in item}
                records.append(kept | stamped)
            offset += len(items)
    return records
