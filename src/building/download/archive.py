"""Where an archive offers a product, and how its files reach the cache."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import httpx

from common.fetch import http, ode, ranges

# How long to wait for the larger half of a product.
TIMEOUT = 60.0


def query_products(client: httpx.Client, **params: str) -> list[dict]:
    """Ask ODE for every product a query matches, as a list of product entries.

    Args:
        client: The client the query goes over.
        params: The ODE query fields, such as ihid, iid, pt or productid.

    Returns:
        entries: One entry per matching product, empty when nothing matched.

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


def file_fields(entry: dict, field: str = "URL") -> dict[str, str]:
    """Map each data file of one product entry to one of its fields, its URL by default.

    Args:
        entry: One product, as `query_products` returns it.
        field: The field read for each file, such as "URL" or "KBytes".

    Returns:
        fields: That field per file of type "Product", keyed by lowercase filename.
    """
    offered = entry.get("Product_files", {}).get("Product_file", [])
    return {
        str(offer.get("FileName", "")).lower(): str(offer.get(field, ""))
        for offer in (offered if isinstance(offered, list) else [offered])
        if offer.get("Type") == "Product"
    }


def product_urls(
    client: httpx.Client, product_id: str, **params: str
) -> dict[str, str]:
    """Ask ODE for the download URL of each file of one product, keyed by suffix.

    Args:
        client: The client the query goes over.
        product_id: The product to ask about.
        params: The other ODE query fields, such as ihid, iid and pt.

    Returns:
        urls: For each suffix, the file named after the product, or else the only
            file with that suffix. A suffix shared by several other files is left out.
    """
    entries = query_products(client, productid=product_id, **params)
    offered = file_fields(entries[0] if entries else {})
    carriers = Counter(Path(name).suffix for name in offered)
    urls = {}
    for name, url in offered.items():
        path = Path(name)
        if url and (path.stem == product_id.lower() or carriers[path.suffix] == 1):
            urls[path.suffix] = url
    return urls


def download_product(
    client: httpx.Client,
    product_id: str,
    destination: dict[str, Path],
    **params: str,
) -> None:
    """Download the files of one ODE product, asking ODE only when one is missing.

    Args:
        client: The client the query and the downloads go over.
        product_id: The product to download.
        destination: Where each of its files belongs, keyed by suffix.
        params: The other ODE query fields, such as ihid, iid and pt.

    Raises:
        FileNotFoundError: When ODE offers no URL for a missing file.
    """
    if any(not path.exists() for path in destination.values()):
        download_files(
            destination, product_urls(client, product_id, **params), client=client
        )


def download_files(
    destination: dict[str, Path],
    urls: dict[str, str],
    *,
    client: httpx.Client | None = None,
    spans: tuple[tuple[int, int], ...] = (),
    size: int = 0,
    origin: int = 0,
) -> None:
    """Download each file from the URL of its suffix, skipping those already on disk.

    Args:
        destination: Where each file belongs, keyed by suffix.
        urls: Where each file is served from, keyed by the same suffix.
        client: A client whose connections to reuse, or None to open one each.
        spans: The byte ranges to keep when size is set, first and past-the-last.
        size: The size of a sparse copy that holds only the spans, or zero to
            download each file whole.
        origin: The byte of the served file a sparse copy starts at.

    Raises:
        FileNotFoundError: When a missing file has no URL.
    """
    for suffix, path in destination.items():
        if path.exists():
            continue
        if not urls.get(suffix):
            raise FileNotFoundError(f"No {suffix} offered for {path.stem}.")
        if size:
            ranges.patched(
                urls[suffix], path, spans, size, TIMEOUT, client=client, origin=origin
            )
        else:
            http.streamed(urls[suffix], path, TIMEOUT, client=client)
