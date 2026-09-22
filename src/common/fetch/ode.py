"""What ODE is asked at, what every query carries, and what it raises."""

from __future__ import annotations

ODE_BASE_URL = "https://oderest.rsl.wustl.edu/live2/"
ODE_TARGET = "mars"

# What every query asks ODE to answer with.
OUTPUT = {"output": "JSON"}


class ODEError(RuntimeError):
    """Raised when ODE reports an error or a query keeps failing."""
