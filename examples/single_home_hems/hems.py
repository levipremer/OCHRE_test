"""Top-level composition of the single-home device controllers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import pandas as pd

from examples.single_home_hems.single_home_hems.controllers import (
    CompositeController,
    DeviceController,
)
from examples.single_home_hems.single_home_hems.forecast import (
    Forecast,
    PriceSchedule,
    TimeOfUsePrice,
)


def _default_device_controllers() -> tuple[DeviceController, ...]:
    """No commands means native immediate-charge baseline, without optimization."""
    return ()


@dataclass
class HEMSController:
    """Apply any selected subset of independently testable device controllers."""

    device_controllers: tuple[DeviceController, ...] = field(
        default_factory=_default_device_controllers
    )
    price_schedule: PriceSchedule = field(default_factory=TimeOfUsePrice)

    def electricity_price_for_step(self, timestamp: pd.Timestamp) -> float:
        """Return the active electricity price in dollars per kWh."""
        return self.price_schedule.price_at(timestamp)

    def controls_for_step(
        self,
        timestamp: pd.Timestamp,
        previous_status: Mapping[str, Any] | None,
        current_schedule: Mapping[str, Any],
        forecast: Forecast,
    ) -> dict[str, dict[str, Any]]:
        """Return the merged controls from the selected device controllers."""
        return CompositeController(self.device_controllers).controls_for_step(
            timestamp,
            previous_status,
            current_schedule,
            forecast,
        )
