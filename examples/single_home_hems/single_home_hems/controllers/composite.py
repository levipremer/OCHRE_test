"""Composition of independently developed device controllers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from ..forecast import Forecast
from .base import DeviceController


@dataclass(frozen=True)
class CompositeController:
    """Merge commands from controllers that own distinct equipment."""

    controllers: tuple[DeviceController, ...] = ()

    def controls_for_step(
        self,
        timestamp: pd.Timestamp,
        previous_status: Mapping[str, Any] | None,
        current_schedule: Mapping[str, Any],
        forecast: Forecast,
    ) -> dict[str, dict[str, Any]]:
        """Evaluate every controller and reject overlapping device ownership."""
        merged: dict[str, dict[str, Any]] = {}
        for controller in self.controllers:
            device_controls = controller.controls_for_step(
                timestamp,
                previous_status,
                current_schedule,
                forecast,
            )
            overlap = merged.keys() & device_controls.keys()
            if overlap:
                raise ValueError(
                    f"Multiple controllers issued commands for {sorted(overlap)}"
                )
            merged.update(device_controls)
        return merged
