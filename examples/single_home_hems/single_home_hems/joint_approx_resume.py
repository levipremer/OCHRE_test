"""Explicitly approximate continuation of a failed joint HEMS replay.

The original run did not checkpoint all OCHRE internal states. Replaying its
minute-level setpoints from a clean start reconstructs the plant, while the
controller's hidden command memory is seeded from observed history. A join
audit makes any discrepancy visible before optimization continues.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


CONTROL_COLUMNS = (
    "HVAC Heating Setpoint (C)",
    "Water Heating Deadband Upper Limit (C)",
    "EV_01A Requested Power (kW)",
)
AUDIT_COLUMNS = (
    "Temperature - Indoor (C)",
    "water_heater_sensor_node_03_temperature_c",
    "water_heater_sensor_node_10_temperature_c",
    "EV_01A End SOC (-)",
    "EV_01A End Stored Energy (kWh)",
    "Total Electric Power (kW)",
)


def read_partial_raw(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    if "Time" not in raw or not set(CONTROL_COLUMNS).issubset(raw.columns):
        raise ValueError(f"Partial OCHRE output lacks required replay controls: {path}")
    times = raw.pop("Time")
    raw.index = pd.DatetimeIndex(pd.to_datetime(times, utc=True).dt.tz_convert(
        pd.Timestamp(times.iloc[0]).tzinfo,
    ), name="Time")
    if raw.index.has_duplicates or not raw.index.is_monotonic_increasing:
        raise ValueError("Partial OCHRE output times must be unique and increasing")
    expected = pd.date_range(raw.index[0], raw.index[-1], freq="min")
    if not raw.index.equals(expected):
        raise ValueError("Partial OCHRE output has a missing minute")
    return raw


class JointApproximateReplay:
    """Replay measured commands, then hand the reconstructed state to MPC."""

    def __init__(self, controller: Any, original_raw: pd.DataFrame,
                 source_directory: Path, audit_path: Path):
        self.controller = controller
        self.original_controls = controller.controls_for_step
        self.original_raw = original_raw
        self.source_directory = Path(source_directory).resolve()
        self.audit_path = Path(audit_path)
        self.join_time = original_raw.index[-1] + pd.Timedelta(minutes=1)
        if self.join_time != self.join_time.floor("15min"):
            raise ValueError("Partial output must end just before a control boundary")
        self.replayed_minutes = 0
        self.join_audited = False

    def _observe_recorded_step(self, timestamp: pd.Timestamp,
                               previous_status: Mapping[str, Any] | None,
                               forecast: Any) -> None:
        if previous_status is None:
            return
        controller = self.controller
        hvac = controller.hvac_controller
        water = controller.water_heater_controller
        hvac.record_warmup_step(timestamp, previous_status, forecast)
        water._observe_heat_pump(timestamp, previous_status)
        model = water._load_model()
        if model is not None:
            measurement = water._measurement(previous_status, model["output_names"])
            if measurement is not None:
                water._observe_legionella_temperature(timestamp, measurement)
        controller._observe_ev(timestamp, previous_status, forecast)

    def _audit_join(self, previous_status: Mapping[str, Any] | None) -> None:
        original = self.original_raw.iloc[-1]
        comparisons = {}
        for name in AUDIT_COLUMNS:
            if previous_status is None or name not in previous_status or name not in original:
                continue
            expected, replayed = float(original[name]), float(previous_status[name])
            if np.isfinite([expected, replayed]).all():
                comparisons[name] = {
                    "original": expected,
                    "replayed": replayed,
                    "absolute_difference": abs(replayed - expected),
                }
        audit = {
            "continuation_type": "approximate_control_replay",
            "original_partial_directory": str(self.source_directory),
            "original_last_complete_minute": str(self.original_raw.index[-1]),
            "first_new_optimization_issue": str(self.join_time),
            "replayed_control_minutes": self.replayed_minutes,
            "controller_state_rule": (
                "Reconstruct HVAC model and comfort history, HPWH observed mode and "
                "legionella memory, and EV measured history from replayed plant steps; "
                "seed held commands from original recorded setpoints. Earlier open-loop "
                "plans and internal solver caches are unavailable."
            ),
            "state_at_join": comparisons,
            "max_audited_absolute_difference": max(
                (item["absolute_difference"] for item in comparisons.values()), default=None,
            ),
        }
        self.audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
        self.join_audited = True

    def controls_for_step(self, timestamp: pd.Timestamp,
                          previous_status: Mapping[str, Any] | None,
                          current_schedule: Mapping[str, Any], forecast: Any):
        timestamp = pd.Timestamp(timestamp)
        if timestamp >= self.join_time:
            if not self.join_audited:
                self._audit_join(previous_status)
            return self.original_controls(timestamp, previous_status, current_schedule, forecast)
        row = self.original_raw.loc[timestamp]
        self._observe_recorded_step(timestamp, previous_status, forecast)
        heating = float(row["HVAC Heating Setpoint (C)"])
        water_setpoint = float(row["Water Heating Deadband Upper Limit (C)"])
        ev_requested = float(row["EV_01A Requested Power (kW)"])
        hvac = self.controller.hvac_controller
        water = self.controller.water_heater_controller
        ev = self.controller.ev_controller
        hvac._last_setpoint = heating
        water._last_setpoint = water_setpoint
        ev.target = ev_requested
        ev.held_mode = int(ev_requested > 1e-7) - int(ev_requested < -1e-7)
        ev.fallback = False
        if timestamp == timestamp.floor(hvac.model_timestep):
            hvac._last_decision_time = timestamp
            water._last_decision_time = timestamp
            ev.next_solve = timestamp + pd.Timedelta(minutes=ev.options.control_minutes)
        self.replayed_minutes += 1
        return {
            hvac.heating_name: {"Setpoint": heating},
            water.equipment_name: {"Setpoint": water_setpoint},
            ev.equipment_name: {
                "P Setpoint": ev_requested,
                "Backed Load (kW)": max(-ev_requested, 0.0),
            },
        }


def complete_replay_audit(output_directory: Path) -> dict[str, Any]:
    """Check all replayed minutes and the stitched 15-minute issue history."""
    output_directory = Path(output_directory)
    audit_path = output_directory / "resume_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    source = Path(audit["original_partial_directory"])
    original = read_partial_raw(source / "data/raw/home_01.csv")
    continued = read_partial_raw(output_directory / "data/raw/home_01.csv")
    replay = continued.reindex(original.index)
    if replay.isna().all(axis=1).any():
        raise ValueError("Completed continuation misses a replayed source minute")
    fields = {}
    for name in AUDIT_COLUMNS:
        if name not in original or name not in replay:
            raise ValueError(f"Replay fidelity audit misses {name}")
        error = replay[name].to_numpy(dtype=float) - original[name].to_numpy(dtype=float)
        audit.setdefault("state_at_join", {})[name] = {
            "original": float(original[name].iloc[-1]),
            "replayed": float(replay[name].iloc[-1]),
            "absolute_difference": float(abs(error[-1])),
        }
        fields[name] = {
            "max_absolute_difference": float(np.max(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(np.square(error)))),
            "difference_at_join": float(error[-1]),
        }
    audit["max_audited_absolute_difference"] = max(
        item["absolute_difference"] for item in audit["state_at_join"].values()
    )
    events = [json.loads(line) for line in
              (output_directory / "whole_home_joint_solver_events.jsonl").read_text().splitlines()]
    issues = pd.DatetimeIndex([pd.Timestamp(row["timestamp"]) for row in events])
    expected = pd.date_range(issues[0], periods=672, freq="15min")
    audit["full_replay_fidelity"] = {
        "replayed_minute_count": len(original),
        "controlled_replayed_minute_count": int(audit["replayed_control_minutes"]),
        "first_replayed_minute": str(original.index[0]),
        "last_replayed_minute": str(original.index[-1]),
        "fields": fields,
    }
    power_error = (replay["Total Electric Power (kW)"]
                   - original["Total Electric Power (kW)"])
    audit["full_replay_fidelity"]["power_integral"] = {
        "signed_energy_difference_kwh": float(power_error.sum() / 60),
        "largest_15_minute_mean_difference_kw": float(
            power_error.resample("15min").mean().abs().max()
        ),
        "minute_count_over_0p1_kw": int(power_error.abs().gt(0.1).sum()),
        "minute_count_over_1_kw": int(power_error.abs().gt(1).sum()),
    }
    audit["solver_event_continuity"] = {
        "event_count": len(events),
        "unique_issue_count": int(issues.nunique()),
        "expected_issue_count": len(expected),
        "strict_15_minute_grid": bool(issues.equals(expected)),
        "join_issue": str(audit["first_new_optimization_issue"]),
        "join_issue_event_count": int((issues == pd.Timestamp(audit["first_new_optimization_issue"])).sum()),
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit
