"""Predictive electric-vehicle controller template."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from ..forecast import Forecast


@dataclass
class EVPredictiveController:
    """Starter EV controller that blocks charging during the TOU peak."""

    ev_names: tuple[str, ...] = ("EV_01A",)
    horizon: pd.Timedelta = pd.Timedelta(hours=18)
    no_charge_price: float = 0.30

    def controls_for_step(
        self,
        timestamp: pd.Timestamp,
        previous_status: Mapping[str, Any] | None,
        current_schedule: Mapping[str, Any],
        forecast: Forecast,
    ) -> dict[str, dict[str, Any]]:
        """Block peak charging; replace this with predictive scheduling logic."""
        # TODO: forecast.window(timestamp, self.horizon) provides prices through
        # the planning horizon. previous_status contains SOC and home/away status.
        if forecast.price_at(timestamp) >= self.no_charge_price:
            return {name: {"Load Fraction": 0} for name in self.ev_names}
        return {}
