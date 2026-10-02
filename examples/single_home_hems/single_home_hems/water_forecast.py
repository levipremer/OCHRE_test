"""Historical water draws and causal scenario forecasts for the water-heater MPC.

Source dates may differ from the simulation dates, but local clock time is
preserved. Morning-conditioning weights belong to the forecast construction;
the optimization treats every retained historical path as equally probable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..water_usage_forecasting.data import FLOW_COLUMN, aggregate_flow
from ..water_usage_forecasting.rf_shortlead import (
    RFShortLeadConfig, apply_rf_shortlead, fit_rf_shortlead,
)
from ..water_usage_forecasting.scenario_methods import (
    ScenarioConfig, ScenarioSet, generate_scenarios,
)


SCENARIO_METHODS = ("daily_budget_high_rf_1", "daily_budget_high")
DEFAULT_WATER_DATA = (
    Path(__file__).resolve().parents[1]
    / "external_data" / "water_flow_dchouse_l_per_min.csv"
)


def _wall_time(timestamp: pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(timestamp)
    return timestamp.tz_localize(None) if timestamp.tz is not None else timestamp


@dataclass
class DCHouseWaterData:
    """One-minute, interval-start L/min data mapped by local calendar time."""

    flow: pd.Series
    simulation_start: pd.Timestamp
    source_start: pd.Timestamp
    source_path: Path | None = None

    @classmethod
    def from_csv(
        cls,
        path: Path,
        simulation_start: pd.Timestamp,
        source_start: str | pd.Timestamp | None = None,
        *,
        lookback_days: int = 30,
        timezone: str = "America/New_York",
    ) -> "DCHouseWaterData":
        """Read converted L/min, preserving missing readings.

        Here ``lookback_days`` only sets the automatic replay start when
        ``source_start`` is omitted. ScenarioWaterForecast.lookback_days
        controls the historical paths included in the optimization.
        """
        raw = pd.read_csv(path)
        if not {"timestamp", FLOW_COLUMN}.issubset(raw.columns) or raw.empty:
            raise ValueError(f"Water CSV needs timestamp and {FLOW_COLUMN} columns")
        # UTC parsing correctly handles the source's changing -04/-05 offsets.
        index = pd.DatetimeIndex(pd.to_datetime(raw["timestamp"], utc=True)).tz_convert(timezone)
        if index.hasnans or index.has_duplicates or not index.is_monotonic_increasing:
            raise ValueError("Water timestamps must be unique, increasing, and nonmissing")
        if not index.equals(index.floor("min", ambiguous="infer")):
            raise ValueError("Water timestamps must align to whole minutes")
        values = pd.to_numeric(raw[FLOW_COLUMN], errors="raise").to_numpy(dtype=float)
        if np.isinf(values).any() or (values < 0).any():
            raise ValueError("Water flow must be nonnegative and finite or missing")
        flow = pd.Series(values, index=index, name=FLOW_COLUMN).reindex(
            pd.date_range(index[0], index[-1], freq="min", name="timestamp")
        )
        simulation_start = pd.Timestamp(simulation_start)
        simulation_wall = _wall_time(simulation_start)
        if source_start is None:
            source_wall = (
                index[0].tz_localize(None).normalize()
                + pd.Timedelta(days=lookback_days)
                + (simulation_wall - simulation_wall.normalize())
            )
            source_start = source_wall.tz_localize(timezone)
        else:
            source_start = pd.Timestamp(source_start)
            source_start = (
                source_start.tz_localize(timezone) if source_start.tz is None
                else source_start.tz_convert(timezone)
            )
        if source_start.time() != simulation_wall.time():
            raise ValueError("Water source start and simulation start must have the same local time of day")
        if source_start not in flow.index:
            raise ValueError("Water source start is outside the one-minute data grid")
        return cls(flow, simulation_start, source_start, Path(path).resolve())

    def _source_index(self, index: pd.DatetimeIndex) -> pd.DatetimeIndex:
        wall = index.tz_localize(None) if index.tz is not None else index
        mapped_wall = wall + (_wall_time(self.source_start) - _wall_time(self.simulation_start))
        # Fail explicitly rather than shift a nonexistent or ambiguous local time.
        return mapped_wall.tz_localize(self.flow.index.tz, ambiguous="raise", nonexistent="raise")

    def source_time(self, timestamp: pd.Timestamp) -> pd.Timestamp:
        return self._source_index(pd.DatetimeIndex([timestamp]))[0]

    def flow_at(
        self, index: pd.DatetimeIndex, *, timestep: pd.Timedelta | None = None,
    ) -> pd.Series:
        """Return measured interval means, refusing missing or out-of-range draws."""
        if not isinstance(index, pd.DatetimeIndex) or index.has_duplicates or not index.is_monotonic_increasing:
            raise ValueError("Simulation water timestamps must be unique and increasing")
        if index.empty:
            return pd.Series(index=index, dtype=float, name=FLOW_COLUMN)
        timestep = pd.Timedelta(timestep) if timestep is not None else (
            index[1] - index[0] if len(index) > 1 else pd.Timedelta(minutes=1)
        )
        minutes = timestep / pd.Timedelta(minutes=1)
        if not np.isfinite(minutes) or minutes < 1 or minutes != int(minutes):
            raise ValueError("Water draw timesteps must be a whole positive number of minutes")
        if len(index) > 1 and not index.equals(pd.date_range(index[0], periods=len(index), freq=timestep)):
            raise ValueError("Simulation water timestamps must be regularly spaced")
        starts = self._source_index(index)
        offsets = pd.to_timedelta(np.arange(int(minutes)), unit="min")
        expanded = pd.DatetimeIndex(np.repeat(starts.asi8, int(minutes)) + np.tile(offsets.asi8, len(starts)), tz="UTC")
        values = self.flow.reindex(expanded).to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Selected DCHouse simulation window has missing or unavailable readings; choose another source start")
        return pd.Series(values.reshape(len(index), int(minutes)).mean(axis=1), index=index, name=FLOW_COLUMN)

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "source_file": str(self.source_path) if self.source_path is not None else None,
            "units": "L/min",
            "timestamp_position": "interval start",
            "source_timezone": str(self.flow.index.tz),
            "simulation_start": self.simulation_start.isoformat(),
            "source_start": self.source_start.isoformat(),
            "source_available_start": self.flow.index[0].isoformat(),
            "source_available_end_exclusive": (self.flow.index[-1] + pd.Timedelta(minutes=1)).isoformat(),
            "mapping": "same local time of day, fixed calendar-date offset",
            "missing_source_minutes": int(self.flow.isna().sum()),
        }


@dataclass
class ScenarioWaterForecast:
    """Generate 48-step forecasts with a frozen, pre-simulation RF fit.

    All complete paths in the lookback window are retained. Conditioning
    weights are used for the RF's first-step mean projection, while the MPC
    explicitly uses uniform probabilities. Without RF, budget weights are
    diagnostic only under this equal-probability interpretation.
    """

    data: DCHouseWaterData
    method: str = "daily_budget_high_rf_1"
    horizon: pd.Timedelta = pd.Timedelta(hours=12)
    timestep: pd.Timedelta = pd.Timedelta(minutes=15)
    lookback_days: int = 30  # Historical candidate days used by the optimizer.
    rf_config: RFShortLeadConfig = field(default_factory=RFShortLeadConfig)
    diagnostic_history: list[dict[str, object]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.method not in SCENARIO_METHODS:
            raise ValueError(f"Scenario method must be one of {SCENARIO_METHODS}")
        if self.horizon != pd.Timedelta(hours=12) or self.timestep != pd.Timedelta(minutes=15):
            raise ValueError("Water scenarios currently require a 12-hour horizon and 15-minute model steps")
        self.config = ScenarioConfig(
            lookback_days=self.lookback_days,
            morning_policy="remaining_budget_high_only",
        )
        self._interval_flow = aggregate_flow(self.data.flow, 15)
        training = self._interval_flow.loc[
            self._interval_flow.index + self.timestep <= self.data.source_start
        ]
        initial = generate_scenarios(
            training, self.data.source_start,
            pd.date_range(self.data.source_start, periods=48, freq=self.timestep), self.config,
        )
        if len(initial.values) < min(7, self.lookback_days):
            raise ValueError("Water source start needs at least seven complete historical paths (or the full shorter lookback)")
        self._initial_candidate_count = len(initial.values)
        self.rf_model = fit_rf_shortlead(training, self.rf_config) if self.method.endswith("_rf_1") else None

    def scenarios_at(self, timestamp: pd.Timestamp) -> ScenarioSet:
        origin = self.data.source_time(timestamp)
        if origin < self.data.source_start:
            raise ValueError("Scenario forecasts cannot precede the simulation/training boundary")
        if origin.minute % 15 or origin.second or origin.microsecond or origin.nanosecond:
            raise ValueError("Scenario forecast origins must align to local quarter hours")
        history = self._interval_flow.loc[self._interval_flow.index + self.timestep <= origin]
        scenarios = generate_scenarios(
            history, origin, pd.date_range(origin, periods=48, freq=self.timestep), self.config,
        )
        if self.rf_model is not None:
            scenarios = apply_rf_shortlead(scenarios, history, self.rf_model, steps=1)
        diagnostics = dict(scenarios.diagnostics)
        diagnostics.update({
            "simulation_origin": pd.Timestamp(timestamp).isoformat(),
            "scenario_method": self.method,
            "optimization_probability": 1 / len(scenarios.values),
            "optimization_probability_policy": "uniform",
            "conditioning_weights": scenarios.weights.tolist(),
            "candidate_ids": list(scenarios.candidate_ids),
            "first_step_conditioned_mean_l_per_min": float(scenarios.weights @ scenarios.values[:, 0]),
            "first_step_uniform_mean_l_per_min": float(scenarios.values[:, 0].mean()),
        })
        self.diagnostic_history.append(diagnostics)
        return ScenarioSet(scenarios.values, scenarios.weights, scenarios.candidate_ids, diagnostics)

    @property
    def metadata(self) -> dict[str, object]:
        return {
            **self.data.metadata,
            "scenario_method": self.method,
            "horizon_hours": 12,
            "interval_minutes": 15,
            "lookback_days": self.lookback_days,
            "initial_complete_candidate_count": self._initial_candidate_count,
            "optimization_probability_policy": "uniform",
            "conditioning_weight_policy": "75% median morning budget gate, 5% uniform mixture; RF mean projection only",
            "rf_model": self.rf_model.diagnostics if self.rf_model is not None else None,
        }
