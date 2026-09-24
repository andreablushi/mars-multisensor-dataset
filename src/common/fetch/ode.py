"""Where ODE is asked, what every query carries, and the client asking it."""

from __future__ import annotations

from typing import Any

import httpx

from common.fetch import http

ODE_BASE_URL = "https://oderest.rsl.wustl.edu/live2/"
ODE_TARGET = "mars"

# What every query asks ODE to answer with.
OUTPUT = {"output": "JSON"}


class ODEError(RuntimeError):
    """Raised when ODE reports an error or a query keeps failing."""


def fetch_results(
    params: dict[str, str], client: httpx.Client | None = None
) -> dict[str, Any]:
    """Run one ODE query and return its parsed ODEResults payload.

    Args:
        params: Query parameters excluding the output format.
        client: A client whose connections to reuse, or None to ask on its own.

    Returns:
        results: The ODEResults object from the response body.

    Raises:
        ODEError: If ODE reports an error of its own.
        FetchError: If ODE refuses the request, or every attempt fails.
    """

    def accepted(payload: Any) -> dict[str, Any] | None:
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

    return http.fetched_json(
        ODE_BASE_URL, {**OUTPUT, **params}, accepted=accepted, client=client
    )


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

    def close(self) -> None:
        """Close the underlying httpx client."""
        self._client.close()

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
        self.close()
