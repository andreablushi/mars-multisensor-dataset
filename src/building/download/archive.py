"""Where an archive offers a product, and how its files reach the cache."""

from __future__ import annotations

from pathlib import Path

import httpx

from common.fetch import http, ode

# How long to wait for the larger half of a product.
TIMEOUT = 60.0


def query(client: httpx.Client, **params: str) -> list[dict]:
    """Read the products one ODE query names, one entry each.

    Args:
        client: The client whose connections the query is asked over.
        params: What to ask for, such as instrument host, instrument and type.

    Returns:
        entries: One entry per product ODE answers with, empty when it matched none.

    Raises:
        ODEError: When ODE reports an error of its own.
        FetchError: When ODE refuses the query, or every attempt fails.
    """
    results = ode.fetch_results(
        {"query": "product", "results": "f", "target": ode.ODE_TARGET, **params},
        client,
    )
    # ODE answers a query that matched nothing with a sentence, not a product.
    products = results.get("Products", {})
    entries = products.get("Product", []) if isinstance(products, dict) else []
    return entries if isinstance(entries, list) else [entries]


def published(entry: dict, field: str = "URL") -> dict[str, str]:
    """Read one field of every file one ODE product entry offers, its URL by default.

    Args:
        entry: One product, as `query` returns it.
        field: The field each file is read for, such as "URL" or "KBytes".

    Returns:
        fields: That field of each file, keyed by its lowercase filename.
    """
    offered = entry.get("Product_files", {}).get("Product_file", [])
    return {
        str(offer.get("FileName", "")).lower(): str(offer.get(field, ""))
        for offer in (offered if isinstance(offered, list) else [offered])
        if offer.get("Type") == "Product"
    }


def offers(client: httpx.Client, product_id: str, **params: str) -> dict[str, str]:
    """Read where ODE offers each file of one product.

    Args:
        client: The client whose connections the query is asked over.
        product_id: The product to ask about.
        params: What else names it, such as the instrument and its type.

    Returns:
        urls: The download URL of each file suffix.
    """
    entries = query(client, productid=product_id, **params)
    named: dict[str, str] = {}
    only: dict[str, str | None] = {}
    for name, url in published(entries[0] if entries else {}).items():
        path = Path(name)
        if path.stem == product_id.lower():
            named[path.suffix] = url
        else:
            only[path.suffix] = None if path.suffix in only else url
    return {suffix: url for suffix, url in {**only, **named}.items() if url}


def collect(
    client: httpx.Client,
    product_id: str,
    destination: dict[str, Path],
    *,
    spans: tuple[tuple[int, int], ...] = (),
    **params: str,
) -> None:
    """Download whichever halves of one ODE product are not on disk yet.

    Args:
        client: The client whose connections the query is asked over.
        product_id: The product to fetch.
        destination: Where each of its halves belongs, keyed by suffix.
        spans: The first and past-the-last byte of each part, or none for whole.
        params: What names the product to ODE, such as host, instrument and type.

    Raises:
        FileNotFoundError: When ODE offers no download for a missing half.
    """
    if any(not path.exists() for path in destination.values()):
        bring(
            destination,
            offers(client, product_id, **params),
            client=client,
            spans=spans,
        )


def bring(
    destination: dict[str, Path],
    urls: dict[str, str],
    *,
    client: httpx.Client | None = None,
    spans: tuple[tuple[int, int], ...] = (),
) -> None:
    """Stream whichever halves of one product are not on disk yet.

    Args:
        destination: Where each half belongs, keyed by suffix.
        urls: Where each half is served from, keyed by the same suffix.
        client: A client whose connections to reuse, or None to open one each.
        spans: The first and past-the-last byte of each part, or none for whole.

    Raises:
        FileNotFoundError: When a missing half is served from nowhere.
    """
    for suffix, path in destination.items():
        if path.exists():
            continue
        if not urls.get(suffix):
            raise FileNotFoundError(f"No {suffix} offered for {path.stem}.")
        http.streamed(urls[suffix], path, TIMEOUT, client=client, spans=spans)
