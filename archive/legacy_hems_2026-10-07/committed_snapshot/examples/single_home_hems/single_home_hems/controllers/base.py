"""Common interface for one-device predictive controllers."""

from __future__ import annotations

from typing import Any, Mapping, Protocol

import pandas as pd

from ..forecast import Forecast


class DeviceController(Protocol):
    """Controller that owns commands for one device type."""

    def controls_for_step(
        self,
        timestamp: pd.Timestamp,
        previous_status: Mapping[str, Any] | None,
        current_schedule: Mapping[str, Any],
        forecast: Forecast,
    ) -> dict[str, dict[str, Any]]:
        """Return this device's OCHRE controls for the current timestep."""
        ...
