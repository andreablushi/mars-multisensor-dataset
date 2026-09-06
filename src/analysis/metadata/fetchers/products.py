"""Fetching one feature and instrument set's product records, a page at a time."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, TypeAlias

import analysis.metadata.provenance as provenance
from analysis.metadata.ode import ODEClient
from analysis.models.feature import Feature
from analysis.models.instrument import InstrumentSet
from utils.fetch import ode_configs
from utils.fetch.ode_configs import ODEError

# A feature running through every longitude is asked for in two halves, since
# no single box ODE accepts reaches round every longitude at once.
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

Box = tuple[float, float, float, float]
ProductRecord: TypeAlias = dict[str, Any]


def _boxes(feature: Feature) -> tuple[Box, ...]:
    """Return the lat/lon boxes a feature has to be asked for in.

    Args:
        feature: The feature to query.

    Returns:
        One box as (min_lat, max_lat, west_lon, east_lon), or two around a pole.
    """
    if feature.circles_a_pole:
        return tuple(
            (feature.min_lat, feature.max_lat, west, east)
            for west, east in LONGITUDE_HALVES
        )
    return ((feature.min_lat, feature.max_lat, feature.west_lon, feature.east_lon),)


def _params(box: Box, instrument_set: InstrumentSet, loc: str) -> dict[str, str]:
    """Build the shared product query parameters for one box and set.

    Args:
        box: The lat/lon box to ask for.
        instrument_set: The instrument host, instrument, and product type.
        loc: "f" for every footprint overlapping the box, "o" for only those inside.

    Returns:
        The parameter dictionary without a results selector.
    """
    min_lat, max_lat, west_lon, east_lon = box
    params = {
        "query": "product",
        "target": ode_configs.ODE_TARGET,
        "ihid": instrument_set.ihid,
        "iid": instrument_set.iid,
        "pt": instrument_set.pt,
        "minlat": str(min_lat),
        "maxlat": str(max_lat),
        "westernlon": str(west_lon),
        "easternlon": str(east_lon),
        "loc": loc,
    }
    if instrument_set.product_id:
        params["productid"] = instrument_set.product_id
    return params


def _pages(client: ODEClient, params: dict[str, str]) -> Iterator[list[Any]]:
    """Count one box's products, then walk them a page at a time.

    Args:
        client: The ODE client to query with.
        params: The box and instrument parameters to page through.

    Yields:
        Each page's raw product items, until they run out.

    Raises:
        ODEError: When ODE reports no usable count.
    """
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
            return
        yield items
        offset += len(items)


def fetch_products(
    client: ODEClient,
    feature: Feature,
    instrument_set: InstrumentSet,
    loc: str,
) -> list[ProductRecord]:
    """Fetch all product metadata for a feature and instrument set.

    Args:
        client: The ODE client to query with.
        feature: The feature whose box the query is built from.
        instrument_set: The instrument host, instrument, and product type.
        loc: Which products the box returns, recorded with each one.

    Returns:
        One record per distinct product, in the order ODE returned them.
    """
    stamped = provenance.stamp(feature, instrument_set, loc)
    records: list[ProductRecord] = []
    # The two boxes a polar feature is asked in overlap, so a product returns twice
    seen: set[tuple[str, str]] = set()
    for box in _boxes(feature):
        for items in _pages(client, _params(box, instrument_set, loc)):
            for item in items:
                identity = (item["Footprint_C0_geometry"], item["UTC_start_time"])
                if identity in seen:
                    continue
                seen.add(identity)
                kept = {f: item[f] for f in RETAINED_FIELDS if f in item}
                records.append(kept | stamped)
    return records
