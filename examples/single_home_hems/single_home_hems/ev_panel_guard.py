"""Measured, minute-level EV charging shutoff for predictive studies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass
class EVPanelGuard:
    """Block positive EV commands after a measured panel exceedance.

    The guard acts on plant commands after predictive control. It does not add
    a constraint to an optimization problem and leaves negative V2H commands
    alone. A trip clears at the next decision or when measured load falls below
    the restart threshold with room for the requested EV charging power.
    """

    limit_kw: float
    decision_interval: pd.Timedelta = pd.Timedelta(minutes=15)
    restart_threshold_kw: float = 25.0
    latched_until: pd.Timestamp | None = None
    previous_applied_charge: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.limit_kw, bool) or not np.isfinite(self.limit_kw) or self.limit_kw <= 0:
            raise ValueError("EV panel guard limit must be positive and finite")
        if (isinstance(self.restart_threshold_kw, bool)
                or not np.isfinite(self.restart_threshold_kw)
                or self.restart_threshold_kw < 0):
            raise ValueError("EV panel guard restart threshold must be nonnegative and finite")
        self.decision_interval = pd.Timedelta(self.decision_interval)
        if self.decision_interval <= pd.Timedelta(0) or pd.Timedelta(days=1) % self.decision_interval:
            raise ValueError("EV panel guard decision interval must divide one day")

    def _next_decision(self, timestamp: pd.Timestamp) -> pd.Timestamp:
        # UTC arithmetic also works at local clock changes because the study's
        # 15-minute grid divides every time-zone offset used by these runs.
        clock = timestamp.tz_convert("UTC") if timestamp.tzinfo is not None else timestamp
        result = clock.floor(self.decision_interval) + self.decision_interval
        return result.tz_convert(timestamp.tz) if timestamp.tzinfo is not None else result

    def apply(
        self,
        timestamp: pd.Timestamp,
        previous_status: Mapping[str, Any] | None,
        controls: dict[str, dict[str, Any]],
        ev_names: tuple[str, ...],
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        """Return the applied commands and a record of the guard decision."""
        timestamp = pd.Timestamp(timestamp)
        if self.latched_until is not None and timestamp >= self.latched_until:
            self.latched_until = None

        measured_kw = np.nan
        if previous_status is not None:
            if "Total Electric Power (kW)" not in previous_status:
                raise ValueError("EV panel guard requires the last completed total-home power")
            measured_kw = float(previous_status["Total Electric Power (kW)"])
            if not np.isfinite(measured_kw):
                raise ValueError("EV panel guard received nonfinite total-home power")

        proposed_kw = sum(max(0.0, float(controls.get(name, {}).get("P Setpoint", 0.0)))
                          for name in ev_names)
        tripped = bool(
            np.isfinite(measured_kw) and measured_kw > self.limit_kw
            and (proposed_kw > 0 or self.previous_applied_charge)
        )
        if tripped:
            self.latched_until = self._next_decision(timestamp)
        elif (self.latched_until is not None
              and np.isfinite(measured_kw)
              and measured_kw < self.restart_threshold_kw
              and proposed_kw > 0
              and measured_kw + proposed_kw <= self.limit_kw):
            self.latched_until = None

        active = self.latched_until is not None and timestamp < self.latched_until
        applied = controls
        if active and proposed_kw > 0:
            applied = {name: values.copy() for name, values in controls.items()}
            for name in ev_names:
                if float(applied.get(name, {}).get("P Setpoint", 0.0)) > 0:
                    applied[name]["P Setpoint"] = 0.0
        applied_kw = sum(max(0.0, float(applied.get(name, {}).get("P Setpoint", 0.0)))
                         for name in ev_names)
        self.previous_applied_charge = applied_kw > 0
        return applied, {
            "ev_panel_guard_observed_total_kw": measured_kw,
            "ev_panel_guard_tripped": tripped,
            "ev_panel_guard_active": active,
            "ev_panel_guard_proposed_charge_kw": proposed_kw,
            "ev_panel_guard_applied_charge_kw": applied_kw,
            "ev_panel_guard_latched_until": (
                None if self.latched_until is None else str(self.latched_until)
            ),
        }
