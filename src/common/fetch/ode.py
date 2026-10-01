"""Where ODE is asked, what every query carries, and the client asking it."""

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
    """Run one ODE query and return its parsed ODEResults payload.

    Args:
        params: Query parameters excluding the output format.
        client: The client whose connections to reuse.

    Returns:
        results: The ODEResults object from the response body.

    Raises:
        ODEError: If ODE reports an error of its own.
        FetchError: If ODE refuses the request, or every attempt fails.
    """
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


class ODEClient:
    """A retrying reader of the ODE REST GET interface."""

    def __init__(self) -> None:
        """Open the client ODE is asked through."""
        self._client = httpx.Client(verify=http.TLS_CONTEXT)

    def query(self, params: dict[str, str]) -> dict[str, Any]:
        """Run one ODE query over this client's connections.

        Args:
            params: Query parameters excluding the output format.

        Returns:
            results: The ODEResults object from the response body.
        """
        return fetch_results(params, self._client)

    def __enter__(self) -> ODEClient:
        """Enter a context manager.

        Returns:
            client: This client.
        """
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the client on context manager exit.

        Args:
            exc: Unused exception information.
        """
        self._client.close()
