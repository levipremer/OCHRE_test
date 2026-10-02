"""Predictive HVAC controller template."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from ..forecast import Forecast


@dataclass
class HVACPredictiveController:
    """Pass-through template for a forecast-aware HVAC controller."""

    horizon: pd.Timedelta = pd.Timedelta(hours=6)
    heating_name: str = "HVAC Heating"
    cooling_name: str = "HVAC Cooling"

    def controls_for_step(
        self,
        timestamp: pd.Timestamp,
        previous_status: Mapping[str, Any] | None,
        current_schedule: Mapping[str, Any],
        forecast: Forecast,
    ) -> dict[str, dict[str, Any]]:
        """Return HVAC controls; implement predictive logic here."""
        # TODO: Use forecast.window(timestamp, self.horizon), indoor temperature
        # from previous_status, and weather/setpoints from current_schedule.
        # Typical outputs set heating_name or cooling_name to {"Setpoint": value_c}.
        return {}
