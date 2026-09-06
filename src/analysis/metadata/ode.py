"""The client the analysis half asks ODE with."""

from __future__ import annotations

from typing import Any

import httpx

from utils.fetch import http, ode_configs


class ODEClient:
    """A retrying reader of the ODE REST GET interface."""

    def __init__(self) -> None:
        """Open the client ODE is asked through.

        Returns:
            None.
        """
        self._client = httpx.Client()

    def query(self, params: dict[str, str]) -> dict[str, Any]:
        """Run one ODE query and return its parsed ODEResults payload.

        Args:
            params: Query parameters excluding the output format.

        Returns:
            The ODEResults object from the response body.

        Raises:
            ODEError: If ODE reports an error of its own.
            FetchError: If ODE refuses the request, or every attempt fails.
        """

        def accepted(payload: Any) -> dict[str, Any] | None:
            """Return the results one reply carries, or None to ask again.

            Args:
                payload: The parsed response body.

            Returns:
                The ODEResults object, or None when the reply holds none.

            Raises:
                ODEError: When ODE reports an error of its own.
            """
            results = payload.get("ODEResults") if isinstance(payload, dict) else None
            if not isinstance(results, dict):
                return None
            if str(results.get("Status", "")).upper() == "ERROR":
                raise ode_configs.ODEError(
                    str(results.get("Error", "unknown ODE error"))
                )
            return results

        return http.fetched_json(
            ode_configs.ODE_BASE_URL,
            {**ode_configs.OUTPUT, **params},
            accepted=accepted,
            client=self._client,
        )

    def close(self) -> None:
        """Close the underlying httpx client.

        Returns:
            None.
        """
        self._client.close()

    def __enter__(self) -> ODEClient:
        """Enter a context manager.

        Returns:
            This client.
        """
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the client on context manager exit.

        Args:
            exc: Unused exception information.

        Returns:
            None.
        """
        self.close()
