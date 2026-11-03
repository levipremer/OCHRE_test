"""Explicit perfect exogenous inputs for offline joint-HEMS comparisons.

Only the fixed split-Hub replay and tempered fixture draw are read from the
future. HVAC electric resistance, compressor/fan power, and HPWH power remain
controller decisions and are never copied from a native thermal trajectory.
The existing water-heater preparation still applies its RC draw conversion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..water_usage_forecasting.scenario_methods import ScenarioSet
from .ev_study_forecasts import PerfectMobilityEVForecast
from .water_forecast import DCHouseWaterData


MINUTE = pd.Timedelta(minutes=1)
INTERVAL = pd.Timedelta(minutes=15)
FIXED_TARGETS = ("hub_other_power_kw", "outside_hub_power_kw")
ORACLE_SOURCE = "perfect_fixed_exogenous_truth_offline_oracle"


def _issue_time(timestamp: pd.Timestamp, index: pd.DatetimeIndex) -> pd.Timestamp:
    timestamp = pd.Timestamp(timestamp)
    if pd.isna(timestamp) or str(timestamp.tz) != str(index.tz):
        raise ValueError("Perfect joint forecast issue and truth must use the same clock/timezone")
    if timestamp.value % INTERVAL.value:
        raise ValueError("Perfect joint forecast issues must align to 15-minute intervals")
    return timestamp


def _steps(steps: int) -> int:
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError("Perfect joint forecast steps must be a positive integer")
    return int(steps)


class PerfectFixedLoadForecast:
    """Exact means of one fixed, interval-start minute replay channel.

    The target whitelist intentionally excludes aggregate Home Hub power and
    Outside-Hub-plus-resistance power: both would freeze thermal decisions in
    a joint controller. Additional thermal columns in an input frame are
    ignored; only the named fixed split replay channel is retained.
    """

    method_label = ORACLE_SOURCE

    def __init__(self, minute_truth: pd.DataFrame, target: str):
        if target not in FIXED_TARGETS:
            raise ValueError(f"Joint fixed-load target must be one of {FIXED_TARGETS}")
        if not isinstance(minute_truth, pd.DataFrame) or target not in minute_truth:
            raise ValueError(f"Joint fixed replay requires {target}")
        index = minute_truth.index
        if (not isinstance(index, pd.DatetimeIndex) or index.empty or index.hasnans
                or index.has_duplicates or not index.is_monotonic_increasing
                or (index.asi8 % MINUTE.value).any()
                or (np.diff(index.asi8) != MINUTE.value).any()):
            raise ValueError("Joint fixed replay needs unique consecutive interval-start minute timestamps")
        values = pd.to_numeric(minute_truth[target], errors="raise").astype(float)
        if not np.isfinite(values).all() or values.lt(0).any():
            raise ValueError("Joint fixed replay power must be finite and nonnegative")
        self.target = target
        self.minute_truth = values.copy()
        self.history: list[dict[str, object]] = []
        self.metadata = {
            "method_label": self.method_label,
            "perfect_forecast": True,
            "information_policy": "offline_oracle_future_fixed_replay_truth",
            "target": target,
            "target_excludes_controlled_hvac_and_hpwh": True,
            "target_excludes_hvac_electric_resistance": True,
            "truth_start": index[0].isoformat(),
            "truth_end_exclusive": (index[-1] + MINUTE).isoformat(),
            "truth_interval_label": "minute_start",
            "aggregation": "arithmetic mean of all fifteen replay minute powers",
        }

    def forecast_kw(self, timestamp: pd.Timestamp, steps: int = 32) -> pd.Series:
        timestamp = _issue_time(timestamp, self.minute_truth.index)
        steps = _steps(steps)
        index = pd.date_range(timestamp, periods=steps * 15, freq=MINUTE)
        rows = self.minute_truth.reindex(index)
        if rows.isna().any():
            raise ValueError("Perfect joint fixed replay lacks complete requested horizon coverage")
        means = rows.to_numpy().reshape(steps, 15).mean(axis=1)
        result = pd.Series(means, index=pd.date_range(timestamp, periods=steps, freq=INTERVAL),
                           name=f"{self.target.removesuffix('_power_kw')}_forecast_kw")
        result.attrs.update(
            source=self.method_label, perfect_forecast=True,
            information_policy=self.metadata["information_policy"],
            target=self.target, target_excludes_controlled_hvac_and_hpwh=True,
            actual_issue_time=timestamp.isoformat(), prior_issue_fallback=False,
            truth_end_exclusive=timestamp + steps * INTERVAL,
        )
        self.history.append({
            "timestamp": timestamp, "issued_at": timestamp,
            "source": self.method_label, "perfect_forecast": True,
            "target": self.target, "steps": steps,
            "truth_end_exclusive": timestamp + steps * INTERVAL,
        })
        return result


@dataclass
class PerfectWaterForecast:
    """One exact fixture-demand scenario through existing HPWH preparation.

    ``DCHouseWaterData.flow_at`` is the plant's replay mapper. Requesting its
    interval mean preserves every minute's volume and the fixed local-calendar
    mapping, while the existing RC routine retains its mains-temperature and
    measured-upper-tank tempered-draw conversion. No future tank state or HPWH
    electrical trajectory is supplied here.
    """

    data: DCHouseWaterData
    horizon: pd.Timedelta = pd.Timedelta(hours=8)
    timestep: pd.Timedelta = INTERVAL
    method: str = field(default="perfect_actual_fixture_draw_offline_oracle", init=False)
    diagnostic_history: list[dict[str, object]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.horizon = pd.Timedelta(self.horizon)
        self.timestep = pd.Timedelta(self.timestep)
        if (pd.isna(self.horizon) or self.horizon <= pd.Timedelta(0)
                or self.timestep != INTERVAL or self.horizon % INTERVAL):
            raise ValueError("Perfect water forecasts require a positive whole-step horizon and 15-minute steps")

    def scenarios_at(self, timestamp: pd.Timestamp) -> ScenarioSet:
        timestamp = _issue_time(timestamp, pd.DatetimeIndex([self.data.simulation_start]))
        if timestamp < self.data.simulation_start:
            raise ValueError("Perfect water forecasts cannot precede the simulation boundary")
        index = pd.date_range(timestamp, periods=int(self.horizon / self.timestep), freq=self.timestep)
        draws = self.data.flow_at(index, timestep=self.timestep).to_numpy(dtype=float)
        diagnostics = {
            "origin": self.data.source_time(timestamp).isoformat(),
            "simulation_origin": timestamp.isoformat(),
            "scenario_method": self.method,
            "perfect_forecast": True,
            "information_policy": "offline_oracle_future_actual_fixture_draw",
            "scenario_count": 1,
            "optimization_probability": 1.0,
            "optimization_probability_policy": "uniform",
            "conditioning_weights": [1.0],
            "candidate_ids": ["perfect_actual_fixture_draw"],
            "first_step_conditioned_mean_l_per_min": float(draws[0]),
            "first_step_uniform_mean_l_per_min": float(draws[0]),
            "truth_end_exclusive": (timestamp + self.horizon).isoformat(),
            "draw_semantics": "tempered fixture-demand interval mean before existing RC conversion",
        }
        self.diagnostic_history.append(diagnostics)
        return ScenarioSet(draws[np.newaxis, :], np.ones(1), ("perfect_actual_fixture_draw",), diagnostics)

    @property
    def metadata(self) -> dict[str, object]:
        return {
            **self.data.metadata,
            "scenario_method": self.method,
            "perfect_forecast": True,
            "information_policy": "offline_oracle_future_actual_fixture_draw",
            "horizon_hours": self.horizon.total_seconds() / 3600,
            "interval_minutes": 15,
            "requested_scenario_count": 1,
            "optimization_probability_policy": "uniform",
            "optimization_probability": 1.0,
            "conditioning_weight_policy": "single actual path, probability one; no RF correction",
            "draw_semantics": "tempered fixture-demand interval mean before existing RC conversion",
            "future_controlled_thermal_states_or_powers_used": False,
        }
