"""Where ODE is asked, what every query carries, and what it answers."""

from __future__ import annotations

from typing import Any

import httpx

from common.fetch import http

ODE_BASE_URL = "https://oderest.rsl.wustl.edu/live2/"
ODE_TARGET = "mars"
PRODUCT_QUERY = {"query": "product", "target": ODE_TARGET}

# What every query asks ODE to answer with.
OUTPUT = {"output": "JSON"}


class ODEError(RuntimeError):
    """Raised when ODE reports an error or a query keeps failing."""


def fetch_results(params: dict[str, str], client: httpx.Client) -> dict[str, Any]:
    """Return the ODEResults of one query, whose params leave out the output format."""
    return http.fetched_json(
        ODE_BASE_URL, {**OUTPUT, **params}, accepted=ode_results, client=client
    )


def ode_results(payload: Any) -> dict[str, Any] | None:
    """Return the results one reply carries, or None to ask again.

    Args:
        payload: The parsed response body.

    Returns:
        results: The ODEResults object, or None when the reply holds none.

    Raises:
        ODEError: When ODE reports an error of its own.
    """
    results = payload.get("ODEResults") if isinstance(payload, dict) else None
    if not isinstance(results, dict):
        return None
    if str(results.get("Status", "")).upper() == "ERROR":
        raise ODEError(str(results.get("Error", "unknown ODE error")))
    return results
