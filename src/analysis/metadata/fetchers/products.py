"""One group and instrument set's product records, fetched a page at a time."""

from __future__ import annotations

from typing import Any

from analysis.models.instrument import InstrumentSet
from analysis.models.tile_group import TileGroup
from common.fetch import ode
from common.fetch.ode import ODEClient, ODEError

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
    "Footprint_NP_geometry",
    "Footprint_SP_geometry",
)


def fetch_products(
    client: ODEClient,
    group: TileGroup,
    instrument_set: InstrumentSet,
    loc: str,
) -> list[dict[str, Any]]:
    """Fetch all product metadata for a group and instrument set.

    Args:
        client: The ODE client to query with.
        group: The group whose box the query is built from.
        instrument_set: The instrument host, instrument, and product type.
        loc: Which products the box returns.

    Returns:
        products: One record per distinct product, in the order ODE returned them.

    Raises:
        ODEError: When ODE reports no usable count for a box.
    """
    records: list[dict[str, Any]] = []
    # The two boxes a polar group is asked in overlap, so a product returns twice
    seen: set[tuple[str, str]] = set()
    spans = (
        LONGITUDE_HALVES
        if group.circles_a_pole
        else ((group.west_lon, group.east_lon),)
    )
    for west_lon, east_lon in spans:
        params = {
            **product_params(instrument_set, instrument_set.pt),
            "minlat": str(group.min_lat),
            "maxlat": str(group.max_lat),
            "westernlon": str(west_lon),
            "easternlon": str(east_lon),
            "loc": loc,
        }
        if instrument_set.product_id:
            params["productid"] = instrument_set.product_id
        for item in every_product(client, params, "opm"):
            identity = (item["Footprint_C0_geometry"], item["UTC_start_time"])
            if identity in seen:
                continue
            seen.add(identity)
            kept = {field: item[field] for field in RETAINED_FIELDS if field in item}
            records.append(kept | {"instrument_set": instrument_set.key})
    return records


def product_params(instrument_set: InstrumentSet, pt: str) -> dict[str, str]:
    """Return what names one instrument's products of one type to ODE.

    Args:
        instrument_set: The instrument host and instrument asked about.
        pt: The product type asked for.

    Returns:
        params: The query parameters naming them.
    """
    return {
        "query": "product",
        "target": ode.ODE_TARGET,
        "ihid": instrument_set.ihid,
        "iid": instrument_set.iid,
        "pt": pt,
    }


def every_product(
    client: ODEClient, params: dict[str, str], results: str
) -> list[dict[str, Any]]:
    """Fetch every product one query matches, a page at a time.

    Args:
        client: The ODE client to query with.
        params: What names the products, excluding the results asked and the paging.
        results: Which ODE result sections each product carries.

    Returns:
        products: Every product ODE answered with, in the order it returned them.

    Raises:
        ODEError: When ODE reports no usable count for the query.
    """
    raw = client.query({**params, "results": "c"}).get("Count")
    try:
        total = int(raw)
    except (TypeError, ValueError):
        raise ODEError(f"ODE returned no product count, found {raw!r}") from None
    products: list[dict[str, Any]] = []
    while len(products) < total:
        page = client.query(
            {
                **params,
                "results": results,
                "order": PAGE_ORDER,
                "limit": str(PAGE_SIZE),
                "offset": str(len(products)),
            }
        )
        answered = page["Products"]["Product"]
        # A box holding one product is answered with that product, not a list of one
        items = answered if isinstance(answered, list) else [answered]
        # Nothing to advance by would page the same offset forever
        if not items:
            break
        products.extend(items)
    return products
