"""Interface between a HEMS controller and the OCHRE dwelling."""

from __future__ import annotations

from typing import Any, Mapping, Protocol

import pandas as pd

from .forecast import Forecast


class HEMSControllerProtocol(Protocol):
    """Control interface called once before each OCHRE model update."""

    def electricity_price_for_step(self, timestamp: pd.Timestamp) -> float:
        """Return the electricity price in dollars per kWh."""
        ...

    def controls_for_step(
        self,
        timestamp: pd.Timestamp,
        previous_status: Mapping[str, Any] | None,
        current_schedule: Mapping[str, Any],
        forecast: Forecast,
    ) -> dict[str, dict[str, Any]]:
        """Return OCHRE equipment controls for the current timestep."""
        ...
