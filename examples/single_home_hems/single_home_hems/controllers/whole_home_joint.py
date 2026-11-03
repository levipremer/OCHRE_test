"""One MILP for the existing HVAC, HPWH, and scenario EV formulations.

The thermal and EV builders retain their own dynamics, costs, and scenario
sharing. This coordinator adds one home-peak epigraph over every paired
water-use and EV-mobility scenario and makes one Gurobi call.
"""

from __future__ import annotations

import gc
import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import cvxpy as cp
import gurobipy as gp
import numpy as np
import pandas as pd

from examples.five_home_block.five_home_block.ev_degradation import turning_points

from ..control_forecasts import EVScenarioSet
from ..ev_home_hub_trace import predicted_trip_energy_fields
from ..forecast import Forecast
from ..rates import (
    LOCAL_TIMEZONE, WinterTOURate, WINTER_TOU, WINTER_TOU_DEMAND,
    WINTER_TOU_DEMAND_CPP, MISO_RTP_DOE,
    GRID2_ON_PEAK_DEMAND_USD_PER_KW_DAY,
    GRID2_ANYTIME_DEMAND_USD_PER_KW_DAY,
)
from ..water_heater_rc import rc_step_matrices
from .fast_joint_milp import ReusableJointMILPSolver, UnsupportedFastMILP
from .ev import EVScenarioController
from .ev_scenario import build_joint_ev_scenario_problem
from .joint_hems import JointHEMSPredictiveController, PEAK_WINDOW_HOURS
from .solver_diagnostics import collect_primal_diagnostics


WHOLE_HOME_WINDOWS = {**PEAK_WINDOW_HOURS, "evening": (16, 22)}
WINTER_TOU_PEAK_WINDOW = "winter-tou"
DOE_PANEL_PENALTY_USD_PER_KWH = 1000.0
DOE_GRID_PENALTY_USD_PER_KWH = 10.0
WLS_REFRESH_SECONDS = 3600.0
WLS_DNS_RETRY_DELAYS_SECONDS = (1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 45.0, 60.0)


def _trace_json_value(value: Any) -> Any:
    """Encode numeric decision inputs without nonstandard JSON NaN values."""
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return value.total_seconds()
    if isinstance(value, pd.Index):
        return _trace_json_value(value.tolist())
    if isinstance(value, np.ndarray):
        return _trace_json_value(value.tolist())
    if isinstance(value, np.generic):
        return _trace_json_value(value.item())
    if isinstance(value, Mapping):
        return {str(key): _trace_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_trace_json_value(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _candidate_numerical_flags(objective, components, scenario_plans):
    """Identify strict audit findings without changing the acceptance gate."""
    flags = []
    balance = float(objective) - sum(float(value) for value in components.values())
    if abs(balance) > 1e-6:
        flags.append(dict(kind="objective_component_balance", residual=balance))
    for index, path in enumerate(scenario_plans):
        expression = path.get("trip_shortfall_kwh")
        if expression is None or expression.value is None:
            continue
        shortfall = np.asarray(expression.value, dtype=float).reshape(-1)
        requested = np.asarray(path["horizon"].trip_kwh, dtype=float).reshape(-1)
        # Negative signed events represent observed away charging; only
        # positive trip demand has shortfall slack in the agreed model.
        upper_bound = np.maximum(requested, 0.0)
        if shortfall.shape != requested.shape:
            flags.append(dict(kind="trip_shortfall_shape", scenario_index=index))
        elif not np.isfinite(shortfall).all():
            flags.append(dict(kind="trip_shortfall_nonfinite", scenario_index=index))
        elif np.any(shortfall < -1e-6) or np.any(shortfall > upper_bound + 1e-6):
            flags.append(dict(kind="trip_shortfall_bounds", scenario_index=index,
                              minimum_kwh=float(shortfall.min()),
                              maximum_above_demand_kwh=float((shortfall - upper_bound).max())))
    return flags


def _predicted_hvac_temperature_c(prepared) -> np.ndarray:
    """Reconstruct the solved indoor output from the prepared HVAC inputs."""
    state = np.asarray(prepared.initial_state, dtype=float).reshape(-1)
    states = [state]
    model = prepared.model
    heat = np.asarray(prepared.heat_hvac.value, dtype=float).reshape(-1)
    for step, heat_w in enumerate(heat):
        state = (model["A"] @ state + model["B"][:, 0] * heat_w
                 + model["E"] @ prepared.disturbances[:, step])
        states.append(state)
    return (model["output_matrix"] @ np.column_stack(states))[0]


def _predicted_water_outputs_c(prepared, model_kind: str, timestep: pd.Timedelta) -> list[dict[str, np.ndarray]]:
    """Reconstruct each water path's solved tank outputs at 33 boundaries."""
    model = prepared.model
    heating = np.asarray(prepared.heat.value, dtype=float)
    paths = []
    for scenario, heat_path in enumerate(heating):
        state = np.asarray(prepared.initial_state, dtype=float).reshape(-1)
        states = [state]
        for step, heat_kw in enumerate(heat_path):
            disturbance = prepared.disturbances[scenario, :, step]
            if model_kind == "rc":
                transition, gain, forcing = rc_step_matrices(
                    model["rc"], *disturbance, timestep.total_seconds() / 3600,
                )
                state = transition @ state + gain * heat_kw + forcing
            else:
                state = (model["A"] @ state + model["B"][:, 0] * heat_kw
                         + model["E"] @ disturbance)
            states.append(np.asarray(state, dtype=float).reshape(-1))
        outputs = model["output_matrix"] @ np.column_stack(states)
        paths.append(dict(zip(model["output_names"], outputs)))
    return paths


def _transient_wls_network_error(error: gp.GurobiError) -> bool:
    message = str(error).lower()
    return any(fragment in message for fragment in (
        "token.gurobi.com", "could not resolve host", "failed to connect",
        "network is unreachable", "connection timed out", "wls license",
    ))


def _average_ev_power_on_thermal_grid(
    plan: Mapping[str, Any], start: pd.Timestamp, steps: int, interval: pd.Timedelta,
    *, power_key: str = "ac",
) -> cp.Expression:
    """Integrate EV AC or charge power exactly across mobility-split blocks."""
    boundaries = pd.DatetimeIndex(plan["boundaries"])
    power = plan[power_key]
    if len(boundaries) != power.size + 1:
        raise ValueError("EV power blocks and boundaries have incompatible lengths")
    end = start + steps * interval
    if boundaries[0] != start or boundaries[-1] < end:
        raise ValueError("EV scenario horizon must cover the thermal horizon")
    hours = interval.total_seconds() / 3600
    weights = np.zeros((steps, power.size), dtype=float)
    for index in range(steps):
        left, right = start + index * interval, start + (index + 1) * interval
        overlaps = np.array([
            max(0.0, (min(right, b) - max(left, a)).total_seconds() / 3600)
            for a, b in zip(boundaries[:-1], boundaries[1:])
        ])
        if not np.isclose(overlaps.sum(), hours, rtol=0, atol=1e-9):
            raise ValueError("EV blocks do not completely cover a thermal interval")
        weights[index] = overlaps / hours
    return weights @ power


@dataclass
class WholeHomeJointPredictiveController:
    """Wrap the latest thermal joint controller and EV scenario MPC."""

    joint_controller: JointHEMSPredictiveController
    ev_controller: EVScenarioController
    peak_weight_usd_per_kw: float
    solver_options: Mapping[str, Any] = field(
        default_factory=lambda: {"TimeLimit": 300.0, "MIPGap": 0.05, "Threads": 1}
    )
    peak_window: str = "night"
    peak_limit_kw: float | None = None
    hard_panel_limit_kw: float | None = None
    doe_panel_limit_kw: float | None = None
    doe_grid_limit_kw: float | None = None
    doe_panel_penalty_weight: float = DOE_PANEL_PENALTY_USD_PER_KWH
    doe_grid_penalty_weight: float = DOE_GRID_PENALTY_USD_PER_KWH
    decision_trace_directory: Path | None = None
    progress_callback: Callable[[Mapping[str, Any]], None] | None = field(default=None, repr=False)
    perfect_forecasts: bool = False
    one_shot: bool = False
    allow_unlimited_solver_time: bool = False
    rate_structure: str = "legacy"
    grid2_demand_formulation: str = "weekday_daily_dt_over_n_v1"
    solver_backend: str = "cvxpy"
    peak_arrival_comfort_enabled: bool = False
    _fast_solver: Any = field(default=None, init=False, repr=False)
    _fast_solver_metrics: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    _solve_history: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _objective_component_history: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _optimization_plans: list[pd.DataFrame] = field(default_factory=list, init=False, repr=False)
    _objective_components: dict[str, cp.Expression] = field(default_factory=dict, init=False, repr=False)
    _ev_power_scenarios_kw: cp.Expression | None = field(default=None, init=False, repr=False)
    _modeled_home_power_scenarios_kw: cp.Expression | None = field(default=None, init=False, repr=False)
    _modeled_panel_load_scenarios_kw: cp.Expression | None = field(default=None, init=False, repr=False)
    _doe_panel_slack_kw: cp.Variable | None = field(default=None, init=False, repr=False)
    _doe_grid_slack_kw: cp.Variable | None = field(default=None, init=False, repr=False)
    _doe_grid_active: np.ndarray | None = field(default=None, init=False, repr=False)
    _rest_of_home_forecast_kw: np.ndarray | None = field(default=None, init=False, repr=False)
    _hub_other_forecast_kw: np.ndarray | None = field(default=None, init=False, repr=False)
    _outside_hub_forecast_kw: np.ndarray | None = field(default=None, init=False, repr=False)
    _peak_objective_active: np.ndarray | None = field(default=None, init=False, repr=False)
    _peak_demand_block_starts: tuple[pd.Timestamp, ...] = field(default=(), init=False, repr=False)
    _peak_observed_first_half_kw: float | None = field(default=None, init=False, repr=False)
    _recent_home_meter_kw: deque[tuple[pd.Timestamp, float]] = field(
        default_factory=lambda: deque(maxlen=15), init=False, repr=False,
    )
    _completed_ev_states: deque[tuple[pd.Timestamp, dict[str, Any]]] = field(
        default_factory=lambda: deque(maxlen=1441), init=False, repr=False,
    )
    _daily_demand_meter_kw: deque[tuple[pd.Timestamp, float]] = field(
        default_factory=lambda: deque(maxlen=1440), init=False, repr=False,
    )
    _restored_daily_demand_peaks: dict[str, dict[str, float]] = field(default_factory=dict, init=False, repr=False)
    _daily_demand_plan: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _daily_demand_objective_terms: dict[str, cp.Expression] = field(default_factory=dict, init=False, repr=False)
    _gurobi_environment_refreshed_at: float = field(default_factory=time.perf_counter, init=False, repr=False)
    _gurobi_environment_refresh_count: int = field(default=0, init=False, repr=False)
    _solver_retry_count: int = field(default=0, init=False, repr=False)
    _wls_network_retry_count: int = field(default=0, init=False, repr=False)
    _one_shot_attempted: bool = field(default=False, init=False, repr=False)
    _one_shot_plan: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _one_shot_thermal_index: int | None = field(default=None, init=False, repr=False)
    _one_shot_replay_history: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    solve_durations_s: dict[str, list[float]] = field(
        default_factory=lambda: {"whole_home_joint_milp": []}, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.solver_backend not in {"cvxpy", "sparse_native_v1"}:
            raise ValueError("Unknown whole-home solver backend")
        if self.solver_backend != "cvxpy" and self.rate_structure not in {
                WINTER_TOU, WINTER_TOU_DEMAND_CPP, MISO_RTP_DOE}:
            raise ValueError("Fast native backend is validated only for Grid 1, Grid 2 CPP, and Grid 3 rates")
        if self.rate_structure not in {"legacy", WINTER_TOU, WINTER_TOU_DEMAND,
                                        WINTER_TOU_DEMAND_CPP, MISO_RTP_DOE}:
            raise ValueError("Unknown whole-home rate structure")
        if self.grid2_demand_formulation not in {"weekday_daily_dt_over_n_v1", "legacy_horizon_peak"}:
            raise ValueError("Unknown whole-home Grid 2 demand formulation")
        if self.hard_panel_limit_kw is not None and (
            self.rate_structure not in {WINTER_TOU, WINTER_TOU_DEMAND, WINTER_TOU_DEMAND_CPP}
            or isinstance(self.hard_panel_limit_kw, bool)
            or not np.isfinite(self.hard_panel_limit_kw)
            or self.hard_panel_limit_kw <= 0
        ):
            raise ValueError("Hard whole-home panel cap needs a positive Grid 1 or 2 limit")
        if self.joint_controller.solver != "GUROBI":
            raise ValueError("Whole-home joint control requires Gurobi")
        ev = self.ev_controller.options
        if ev.formulation != "milp" or not ev.v2h:
            raise ValueError("Whole-home joint control requires bidirectional EV MILP")
        if not ev.enforce_terminal_soc or ev.enable_degradation:
            raise ValueError("Whole-home EV needs terminal energy value and no degradation cost")
        if not (np.isclose(ev.electricity_weight, 1) and np.isclose(ev.health_soc_min, 0.2)
                and np.isclose(ev.health_soc_max, 0.8) and np.isclose(ev.health_weight, 1)
                and np.isclose(ev.health_soc_weight, 1000)
                and np.isclose(ev.comfort_soc_min,
                               0.30 if self.peak_arrival_comfort_enabled else 0.40)
                and np.isclose(ev.comfort_weight, 1000) and np.isclose(ev.persistence_weight, 0)
                and not ev.enforce_cycle_limit and not ev.enforce_dwell_constraints
                and not ev.enforce_discharge_budget):
            raise ValueError("Whole-home EV requires the existing milp_health objective variant")
        if self.peak_arrival_comfort_enabled:
            self.ev_controller.configure_observed_peak_comfort(self.rate_structure)
        thermal_step = self.hvac_controller.model_timestep
        if pd.Timedelta(minutes=ev.control_minutes) != thermal_step:
            raise ValueError("EV and thermal controllers must share the 15-minute control grid")
        if self.one_shot and not self.perfect_forecasts:
            raise ValueError("One-shot whole-home control requires explicit perfect forecasts")
        extended_rolling_horizon = (
            self.joint_controller.horizon in {
                pd.Timedelta(hours=16), pd.Timedelta(hours=20), pd.Timedelta(hours=24)
            }
            and self.hvac_controller.full_horizon_rolling_budget
        )
        if (self.joint_controller.horizon <= pd.Timedelta(0)
                or self.joint_controller.horizon % thermal_step
                or self.ev_controller.horizon != self.joint_controller.horizon
                or (not self.one_shot and self.joint_controller.horizon != pd.Timedelta(hours=8)
                    and not extended_rolling_horizon)):
            raise ValueError(
                "Whole-home HVAC, water-heater, and EV horizons must match; "
                "rolling control requires eight hours or an explicitly selected "
                "16/20/24-hour rolling HVAC comfort budget"
            )
        if thermal_step != pd.Timedelta(minutes=15):
            raise ValueError("Whole-home controllers require a 15-minute model timestep")
        if self.peak_window not in (*WHOLE_HOME_WINDOWS, WINTER_TOU_PEAK_WINDOW):
            raise ValueError("Unknown whole-home peak window")
        if self.peak_limit_kw is not None and (
            isinstance(self.peak_limit_kw, bool) or not np.isfinite(self.peak_limit_kw)
            or self.peak_limit_kw <= 0
        ):
            raise ValueError("Whole-home peak limit must be finite and positive")
        if self.doe_panel_limit_kw is not None and (
            isinstance(self.doe_panel_limit_kw, bool) or not np.isfinite(self.doe_panel_limit_kw)
            or self.doe_panel_limit_kw <= 0
        ):
            raise ValueError("Whole-home DOE panel limit must be finite and positive")
        if self.doe_grid_limit_kw is not None and (
            isinstance(self.doe_grid_limit_kw, bool) or not np.isfinite(self.doe_grid_limit_kw)
            or self.doe_grid_limit_kw <= 0
        ):
            raise ValueError("Whole-home DOE grid limit must be finite and positive")
        if any(
            isinstance(weight, bool) or not np.isfinite(weight) or weight <= 0
            for weight in (self.doe_panel_penalty_weight, self.doe_grid_penalty_weight)
        ):
            raise ValueError("Whole-home DOE penalty weights must be finite and positive")
        if not np.isfinite(self.peak_weight_usd_per_kw) or self.peak_weight_usd_per_kw < 0:
            raise ValueError("Whole-home peak weight must be nonnegative and finite")
        if self.uses_daily_grid2_demand:
            # Compatibility field denotes the effective on-peak daily rate;
            # the separate anytime rate and dt/N are logged independently.
            self.peak_weight_usd_per_kw = GRID2_ON_PEAK_DEMAND_USD_PER_KW_DAY
        options = dict(self.solver_options)
        limit = options.get("TimeLimit")
        if limit is None and self.allow_unlimited_solver_time:
            # Gurobi's unlimited default is used only after explicit opt-in.
            options.pop("TimeLimit", None)
        elif (isinstance(limit, bool) or limit is None
                or not np.isfinite(limit) or limit <= 0):
            raise ValueError("Whole-home solve needs a positive finite time limit")
        if not np.isclose(options.get("MIPGap", -1), 0.05):
            raise ValueError("Whole-home solve must use the approved 5% MIP gap")
        self.solver_options = options
        if self.decision_trace_directory is not None:
            directory = Path(self.decision_trace_directory)
            if directory.exists() and any(directory.glob("joint_issue_*.json")):
                raise FileExistsError(
                    f"Decision trace directory already has joint issues: {directory}"
                )
            directory.mkdir(parents=True, exist_ok=True)
            self.decision_trace_directory = directory

    @property
    def water_heater_controller(self):
        return self.joint_controller.water_heater_controller

    @property
    def hvac_controller(self):
        return self.joint_controller.hvac_controller

    @property
    def horizon(self) -> pd.Timedelta:
        return self.joint_controller.horizon

    @property
    def uses_daily_grid2_demand(self) -> bool:
        return (self.rate_structure == WINTER_TOU_DEMAND_CPP
                and self.grid2_demand_formulation == "weekday_daily_dt_over_n_v1")

    @property
    def peak_window_hours(self) -> tuple[int, int]:
        return WHOLE_HOME_WINDOWS[self.peak_window]

    @property
    def solve_history(self) -> pd.DataFrame:
        return pd.DataFrame(self._solve_history)

    @property
    def objective_component_history(self) -> pd.DataFrame:
        return pd.DataFrame(self._objective_component_history)

    @property
    def optimization_plans(self) -> tuple[pd.DataFrame, ...]:
        return tuple(self._optimization_plans)

    @property
    def one_shot_replay_history(self) -> pd.DataFrame:
        return pd.DataFrame(self._one_shot_replay_history)

    @property
    def accepted_one_shot_plan(self) -> Mapping[str, Any] | None:
        return self._one_shot_plan

    def _valid_scenario_counts(self, water_count: int, ev_count: int) -> bool:
        if self.perfect_forecasts:
            return water_count == 1 and ev_count == 1
        from ..water_forecast import RMSE_VOLUME_6H_METHODS
        expected_water_count = RMSE_VOLUME_6H_METHODS.get(
            getattr(self.water_heater_controller, "scenario_method", None), 7)
        return water_count == expected_water_count and ev_count in (5, 7)

    def _notify_progress(self, row: Mapping[str, Any]) -> None:
        if self.progress_callback is None:
            return
        durations = self.solve_durations_s["whole_home_joint_milp"]
        self.progress_callback({
            **row,
            "solve_count": len(self._solve_history),
            "successful_solves": sum(bool(entry["successful"]) for entry in self._solve_history),
            "solve_seconds_total": sum(durations),
            "solve_seconds_max": max(durations, default=0.0),
            "time_limit_hits": sum(bool(entry.get("time_limit_hit")) for entry in self._solve_history),
        })

    def _peak_mask(self, timestamp: pd.Timestamp, steps: int) -> np.ndarray:
        index = pd.date_range(timestamp, periods=steps, freq=self.hvac_controller.model_timestep)
        if self.peak_window == WINTER_TOU_PEAK_WINDOW:
            return np.fromiter(
                (WinterTOURate.is_on_peak(target) for target in index),
                dtype=bool, count=steps,
            )
        first, last = self.peak_window_hours
        return np.asarray((index.hour >= first) & (index.hour < last), dtype=bool)

    def _doe_grid_mask(self, timestamp: pd.Timestamp, steps: int) -> np.ndarray:
        """Scenario 3 weekday 06:00–09:00 and 17:00–21:00 local windows."""
        index = pd.date_range(timestamp, periods=steps, freq=self.hvac_controller.model_timestep)
        if index.tz is not None:
            index = index.tz_convert(LOCAL_TIMEZONE)
        return np.asarray(
            (index.dayofweek < 5)
            & (((index.hour >= 6) & (index.hour < 9))
               | ((index.hour >= 17) & (index.hour < 21))),
            dtype=bool,
        )

    def observe_plant_step(
        self, timestamp: pd.Timestamp, status: Mapping[str, Any] | None,
    ) -> None:
        """Observe HPWH starts and the completed minute's whole-home meter."""
        if self.peak_arrival_comfort_enabled and status is not None:
            name = self.ev_controller.equipment_name
            keys = (f"{name} SOC (-)", f"{name} Capacity (kWh)", f"{name} Parked")
            if not all(key in status for key in keys):
                raise ValueError("Whole-home warm-up lacks measured EV SOC, capacity, or parked state")
            self._completed_ev_states.append((
                pd.Timestamp(timestamp) + pd.Timedelta(minutes=1),
                {key: status[key] for key in keys},
            ))
        observe_starts = getattr(self.water_heater_controller, "observe_start_history", None)
        if status is not None and observe_starts is not None:
            observe_starts(timestamp, status)
        if (not self.uses_daily_grid2_demand
                and (self.peak_window != WINTER_TOU_PEAK_WINDOW or self.peak_weight_usd_per_kw == 0)):
            return
        if status is None or "Total Electric Power (kW)" not in status:
            raise ValueError("Winter demand control requires the whole-home power meter")
        power_kw = float(status["Total Electric Power (kW)"])
        if not np.isfinite(power_kw):
            raise ValueError("Whole-home meter power must be finite")
        self._recent_home_meter_kw.append((pd.Timestamp(timestamp), power_kw))
        if self.uses_daily_grid2_demand:
            self._daily_demand_meter_kw.append((pd.Timestamp(timestamp), power_kw))

    @staticmethod
    def _demand_local_time(timestamp):
        timestamp = pd.Timestamp(timestamp)
        return timestamp if timestamp.tz is None else timestamp.tz_convert(LOCAL_TIMEZONE)

    def _observed_daily_demand_peaks(self, timestamp):
        """Causal complete-block import maxima for the current local weekday."""
        timestamp = pd.Timestamp(timestamp)
        day = self._demand_local_time(timestamp).normalize()
        if day.weekday() >= 5:
            return {}
        peaks = {"on_peak_kw": 0.0, "anytime_kw": 0.0}
        for saved_day, saved in self._restored_daily_demand_peaks.items():
            if self._demand_local_time(saved_day).normalize() == day:
                for key in peaks:
                    value = float(saved[key])
                    if not np.isfinite(value) or value < 0:
                        raise ValueError("Restored daily demand peaks must be finite and nonnegative")
                    peaks[key] = max(peaks[key], value)
        first = getattr(self.ev_controller, "first_observation", None)
        by_block = {}
        for start, value in self._daily_demand_meter_kw:
            if (start + pd.Timedelta(minutes=1) > timestamp
                    or self._demand_local_time(start).normalize() != day
                    or (first is not None and start < first)):
                continue
            by_block.setdefault(start.floor("30min"), []).append((start, value))
        for block_start, readings in by_block.items():
            expected = pd.date_range(block_start, periods=30, freq="min")
            if [start for start, _ in readings] != list(expected):
                continue
            mean_import = max(0.0, float(np.mean([value for _, value in readings])))
            peaks["anytime_kw"] = max(peaks["anytime_kw"], mean_import)
            if WinterTOURate.is_on_peak(block_start):
                peaks["on_peak_kw"] = max(peaks["on_peak_kw"], mean_import)
        return {day.isoformat(): peaks}

    def _build_daily_grid2_demand(self, combined, timestamp, steps, observed_first_half_kw):
        """One epigraph per weekday/component over every existing paired path."""
        interval = self.hvac_controller.model_timestep
        starts = pd.date_range(timestamp, periods=steps, freq=interval)
        blocks, block_starts = [], []
        active = np.zeros(steps, dtype=bool)
        self._peak_observed_first_half_kw = None
        if starts[0].minute % 30 == 15 and self._demand_local_time(starts[0]).weekday() < 5:
            if observed_first_half_kw is None or not np.isfinite(observed_first_half_kw):
                raise ValueError("A weekday mid-block issue needs its observed first-half power")
            self._peak_observed_first_half_kw = float(observed_first_half_kw)
            blocks.append((combined[:, 0] + observed_first_half_kw) / 2)
            block_starts.append(starts[0] - interval)
            active[0] = True
        for step, start in enumerate(starts[:-1]):
            if start.minute % 30 == 0 and self._demand_local_time(start).weekday() < 5:
                blocks.append((combined[:, step] + combined[:, step + 1]) / 2)
                block_starts.append(start)
                active[step:step + 2] = True
        block_starts = tuple(block_starts)
        block_power = cp.vstack(blocks).T if blocks else None
        on_peak = np.array([WinterTOURate.is_on_peak(t) for t in block_starts], dtype=bool)
        days = tuple(dict.fromkeys(self._demand_local_time(t).normalize() for t in starts
                                  if self._demand_local_time(t).weekday() < 5))
        observed = self._observed_daily_demand_peaks(timestamp)
        observed_on = np.array([observed.get(day.isoformat(), {}).get("on_peak_kw", 0.0) for day in days])
        observed_any = np.array([observed.get(day.isoformat(), {}).get("anytime_kw", 0.0) for day in days])
        scale = interval.total_seconds() / 3600 / steps
        constraints = []
        if days:
            on = cp.Variable(len(days), nonneg=True, name="joint_daily_on_peak_demand_kw")
            anytime = cp.Variable(len(days), nonneg=True, name="joint_daily_anytime_demand_kw")
            constraints += [on >= observed_on, anytime >= observed_any]
            for index, day in enumerate(days):
                day_blocks = np.array([self._demand_local_time(t).normalize() == day
                                       for t in block_starts], dtype=bool)
                if day_blocks.any():
                    constraints.append(anytime[index] >= block_power[:, day_blocks])
                if (day_blocks & on_peak).any():
                    constraints.append(on[index] >= block_power[:, day_blocks & on_peak])
            costs = {
                "on_peak": scale * GRID2_ON_PEAK_DEMAND_USD_PER_KW_DAY * cp.sum(on),
                "anytime": scale * GRID2_ANYTIME_DEMAND_USD_PER_KW_DAY * cp.sum(anytime),
            }
        else:
            on, anytime = None, None
            costs = {key: cp.Constant(0.0) for key in ("on_peak", "anytime")}
        self._daily_demand_objective_terms = costs
        self._daily_demand_plan = {
            "daily_demand_days": days,
            "daily_on_peak_demand_kw": on, "daily_anytime_demand_kw": anytime,
            "daily_demand_observed_on_peak_kw": observed_on,
            "daily_demand_observed_anytime_kw": observed_any,
            "daily_demand_observed_peaks": observed,
            "daily_demand_scale": scale, "daily_demand_normalization_steps": steps,
            "daily_demand_step_hours": interval.total_seconds() / 3600,
            "daily_demand_on_peak_rate_usd_per_kw_day": GRID2_ON_PEAK_DEMAND_USD_PER_KW_DAY,
            "daily_demand_anytime_rate_usd_per_kw_day": GRID2_ANYTIME_DEMAND_USD_PER_KW_DAY,
            "daily_demand_on_peak_coefficient_usd_per_kw": scale * GRID2_ON_PEAK_DEMAND_USD_PER_KW_DAY,
            "daily_demand_anytime_coefficient_usd_per_kw": scale * GRID2_ANYTIME_DEMAND_USD_PER_KW_DAY,
            "daily_demand_block_starts": block_starts,
            "daily_demand_block_power_scenarios_kw": block_power,
            "daily_demand_block_is_on_peak": on_peak,
            "daily_demand_active": active,
        }
        self._peak_demand_block_starts = tuple(t for t, selected in zip(block_starts, on_peak) if selected)
        peak_active = np.array([t.floor("30min") in self._peak_demand_block_starts for t in starts])
        return constraints, peak_active

    def daily_demand_metadata(self, *, solved=False):
        """Separate nonadditive diagnostics from the sole total objective term."""
        if not self.uses_daily_grid2_demand:
            return {}
        result = {"grid2_demand_formulation": self.grid2_demand_formulation}
        for key, value in self._daily_demand_plan.items():
            if isinstance(value, cp.Expression):
                value = value.value if solved else None
            result[key] = _trace_json_value(value)
        result["daily_demand_objective_terms"] = {
            key: float(value.value) if solved else None
            for key, value in self._daily_demand_objective_terms.items()
        }
        return result

    def _observed_first_half_kw(self, issue_time: pd.Timestamp) -> float:
        """Mean the fifteen complete meter minutes preceding a mid-block issue."""
        expected = pd.date_range(
            pd.Timestamp(issue_time) - pd.Timedelta(minutes=15),
            periods=15, freq="min",
        )
        observed = tuple(self._recent_home_meter_kw)
        if len(observed) != 15 or any(actual != target for (actual, _), target in zip(observed, expected)):
            raise ValueError(
                "Winter demand control needs 15 consecutive observed whole-home meter "
                "minutes before an on-peak :15 or :45 decision"
            )
        return float(np.mean([power_kw for _, power_kw in observed]))

    def _winter_demand_block_power(
        self, combined: cp.Expression, timestamp: pd.Timestamp, steps: int,
        observed_first_half_kw: float | None,
    ) -> tuple[cp.Expression | None, np.ndarray]:
        """Build aligned 30-minute means from complete predicted or mixed blocks."""
        index = pd.date_range(timestamp, periods=steps, freq=self.hvac_controller.model_timestep)
        used = np.zeros(steps, dtype=bool)
        blocks = []
        starts = []
        self._peak_observed_first_half_kw = None
        if index[0].minute % 30 == 15:
            block_start = index[0] - pd.Timedelta(minutes=15)
            if WinterTOURate.is_on_peak(block_start):
                if observed_first_half_kw is None or not np.isfinite(observed_first_half_kw):
                    raise ValueError("An on-peak mid-block issue needs its observed first-half power")
                self._peak_observed_first_half_kw = float(observed_first_half_kw)
                blocks.append((combined[:, 0] + observed_first_half_kw) / 2)
                starts.append(block_start)
                used[0] = True
        for step in range(steps - 1):
            start = index[step]
            if start.minute % 30 != 0 or not WinterTOURate.is_on_peak(start):
                continue
            if not WinterTOURate.is_on_peak(index[step + 1]):
                continue
            blocks.append((combined[:, step] + combined[:, step + 1]) / 2)
            starts.append(start)
            used[step:step + 2] = True
        self._peak_demand_block_starts = tuple(starts)
        return (cp.vstack(blocks).T if blocks else None), used

    def _eligible_power_upper_kw(self, water, hvac, hub_other_kw=None) -> float:
        """Upper bound on modeled HPWH, HVAC compressor/fan, and Hub-other."""
        heater = self.water_heater_controller
        water_cop = water.model["rc"].cop if heater.model_kind == "rc" else heater.cop
        water_upper_kw = heater._heat_limit_kw(water.model) / water_cop
        hvac_limits = self.hvac_controller._power_and_er_limits_w(hvac.previous_status)
        if hvac_limits is None:
            raise ValueError("HVAC compressor-and-fan power limit is unavailable")
        hub_max = 0.0 if hub_other_kw is None else float(np.max(hub_other_kw))
        upper_kw = float(water_upper_kw + hvac_limits[0] / 1000 + hub_max)
        if not np.isfinite(upper_kw) or upper_kw <= 0:
            raise ValueError("EV-backed thermal power limit must be positive and finite")
        return upper_kw

    def optimization_problem_milp(
        self, water, hvac, ev_problem: cp.Problem, ev_plan: Mapping[str, Any],
        rest_of_home_kw: np.ndarray | None = None, *,
        hub_other_kw: np.ndarray | None = None,
        outside_hub_kw: np.ndarray | None = None,
        observed_first_half_kw: float | None = None,
    ) -> cp.Problem:
        """Combine device terms and the paired whole-home peak exceedance."""
        water_power, hvac_power = water.electric_power_kw, hvac.electric_power_kw
        water_count, steps = water_power.shape
        paths = ev_plan["scenario_plans"]
        ev_count = ev_plan.get("scenario_shape", (None, None))[1]
        if (not self._valid_scenario_counts(water_count, ev_count)
                or len(paths) != water_count * ev_count
                or ev_plan.get("scenario_shape") != (water_count, ev_count)):
            raise ValueError("Whole-home control requires every paired water/EV scenario")
        if any(path.get("water_scenario_index") != water_index or
               path.get("ev_scenario_index") != ev_index
               for index, path in enumerate(paths)
               for water_index, ev_index in [(index // ev_count, index % ev_count)]):
            raise ValueError("EV paths must use water-major, EV-minor pair ordering")
        if hvac_power.shape != (steps,) or steps != int(self.horizon / self.hvac_controller.model_timestep):
            raise ValueError("Thermal power trajectories have incompatible dimensions")
        prices = np.asarray(water.prices, dtype=float).reshape(-1)
        if prices.shape != (steps,) or not np.allclose(prices, hvac.prices, rtol=0, atol=1e-12):
            raise ValueError("HVAC and water-heater price forecasts must agree")
        split = hub_other_kw is not None or outside_hub_kw is not None
        if split:
            if rest_of_home_kw is not None or hub_other_kw is None or outside_hub_kw is None:
                raise ValueError("Use both split Hub forecasts without aggregate rest")
            hub = np.asarray(hub_other_kw, dtype=float).reshape(-1)
            outside = np.asarray(outside_hub_kw, dtype=float).reshape(-1)
            if any(array.shape != (steps,) or not np.isfinite(array).all() or (array < 0).any()
                   for array in (hub, outside)):
                raise ValueError("Split Hub forecasts need finite nonnegative 15-minute power")
            rest = hub + outside
            self._hub_other_forecast_kw = hub.copy()
            self._outside_hub_forecast_kw = outside.copy()
        else:
            rest = np.asarray(rest_of_home_kw, dtype=float).reshape(-1)
            if rest.shape != (steps,) or not np.isfinite(rest).all() or (rest < 0).any():
                raise ValueError("The rest-of-home forecast must contain finite nonnegative power")
            self._hub_other_forecast_kw = None
            self._outside_hub_forecast_kw = None
        self._rest_of_home_forecast_kw = rest.copy()
        thermal_power = water_power + cp.vstack([hvac_power for _ in range(water_count)])
        ev_power = cp.vstack([
            _average_ev_power_on_thermal_grid(p, hvac.timestamp, steps, self.hvac_controller.model_timestep)
            for p in paths
        ])
        self._ev_power_scenarios_kw = ev_power
        # Each EV path is already coupled to its water-use path. HVAC and the
        # background forecasts are common across all water/EV pairs.
        water_indices = np.repeat(np.arange(water_count), ev_count)
        combined = (thermal_power[water_indices, :] + ev_power
                    + np.broadcast_to(rest, (len(paths), steps)))
        self._modeled_home_power_scenarios_kw = combined
        self._modeled_panel_load_scenarios_kw = None
        self._doe_panel_slack_kw = None
        self._doe_grid_slack_kw = None
        self._doe_grid_active = None
        dt_hours = self.hvac_controller.model_timestep.total_seconds() / 3600
        thermal_electricity = dt_hours / (steps * water_count) * cp.sum(cp.multiply(
            np.broadcast_to(prices, (water_count, steps)), thermal_power
        ))
        components: dict[str, cp.Expression] = {"electricity_objective_cost": thermal_electricity}
        for prefix, prepared in (("hpwh", water), ("hvac", hvac)):
            for name, expression in prepared.objective_components.items():
                if name != "electricity_objective_cost":
                    components[f"{prefix}_{name}"] = expression
        for name, expression in ev_plan["objective_terms"].items():
            components[f"ev_{name}_objective_cost"] = expression
        self._daily_demand_plan = {}
        self._daily_demand_objective_terms = {}
        daily_constraints = []
        if self.uses_daily_grid2_demand:
            daily_constraints, active = self._build_daily_grid2_demand(
                combined, hvac.timestamp, steps, observed_first_half_kw,
            )
            peak_power = None
        elif self.peak_window == WINTER_TOU_PEAK_WINDOW and self.peak_weight_usd_per_kw > 0:
            peak_power, active = self._winter_demand_block_power(
                combined, hvac.timestamp, steps, observed_first_half_kw,
            )
        else:
            active = self._peak_mask(hvac.timestamp, steps) & (self.peak_weight_usd_per_kw > 0)
            peak_power = combined[:, active] if active.any() else None
            self._peak_demand_block_starts = ()
            self._peak_observed_first_half_kw = None
        self._peak_objective_active = active
        constraints = [*water.problem.constraints, *hvac.problem.constraints,
                       *ev_problem.constraints, *daily_constraints]
        if self.doe_panel_limit_kw is not None or self.hard_panel_limit_kw is not None:
            # The panel-loading proxy counts all household loads and EV charging.
            # V2H discharge reduces net grid import, but it does not cancel the
            # modeled load supplied through the panel from grid and EV sources.
            ev_charge = cp.vstack([
                _average_ev_power_on_thermal_grid(
                    path, hvac.timestamp, steps, self.hvac_controller.model_timestep,
                    power_key="charge",
                )
                for path in paths
            ])
            panel_load = (thermal_power[water_indices, :] + ev_charge
                          + np.broadcast_to(rest, (len(paths), steps)))
            self._modeled_panel_load_scenarios_kw = panel_load
            if self.hard_panel_limit_kw is not None:
                constraints.append(panel_load <= self.hard_panel_limit_kw)
            if self.doe_panel_limit_kw is not None:
                panel_slack = cp.Variable(steps, nonneg=True, name="doe_panel_slack_kw")
                self._doe_panel_slack_kw = panel_slack
                constraints.append(panel_load <= self.doe_panel_limit_kw + panel_slack)
                components["doe_panel_violation_objective_cost"] = (
                    self.doe_panel_penalty_weight * dt_hours / steps * cp.sum(panel_slack)
                )
        if self.doe_grid_limit_kw is not None:
            grid_active = self._doe_grid_mask(hvac.timestamp, steps)
            self._doe_grid_active = grid_active
            if grid_active.any():
                grid_slack = cp.Variable(
                    int(grid_active.sum()), nonneg=True, name="doe_grid_slack_kw",
                )
                self._doe_grid_slack_kw = grid_slack
                constraints.append(combined[:, grid_active] <= self.doe_grid_limit_kw + grid_slack)
                components["doe_grid_violation_objective_cost"] = (
                    self.doe_grid_penalty_weight * dt_hours / steps * cp.sum(grid_slack)
                )
            else:
                components["doe_grid_violation_objective_cost"] = cp.Constant(0.0)
        if self.uses_daily_grid2_demand:
            components["modeled_home_peak_objective_cost"] = sum(self._daily_demand_objective_terms.values())
        elif peak_power is not None:
            peak_kw = cp.Variable(nonneg=True, name="modeled_home_peak_kw")
            if self.peak_limit_kw is None:
                # Preserve the older raw-peak pilot when no threshold is set.
                constraints.append(peak_kw >= peak_power)
                coefficient = self.peak_weight_usd_per_kw
            else:
                constraints.append(peak_kw >= peak_power - self.peak_limit_kw)
                coefficient = self.peak_weight_usd_per_kw * dt_hours / steps
            components["modeled_home_peak_objective_cost"] = coefficient * peak_kw
        else:
            components["modeled_home_peak_objective_cost"] = cp.Constant(0.0)
        self._objective_components = components
        problem = cp.Problem(cp.Minimize(sum(components.values())), constraints)
        if not problem.is_dcp() or not problem.is_mixed_integer():
            raise ValueError("Whole-home formulation must be a DCP-compliant MILP")
        return problem

    def _observe_ev(self, timestamp: pd.Timestamp, previous_status, forecast: Forecast):
        """Refresh measured EV state once per plant step, as the standalone MPC does."""
        c = self.ev_controller
        status = dict(previous_status or {})
        status.update(getattr(forecast, "ev_states", {}))
        name = c.equipment_name
        soc = float(np.clip(status[f"{name} SOC (-)"], 0, 1))
        step = forecast.data.index[1] - forecast.data.index[0]
        # The preceding plant interval used the previous measured capacity.
        # Preserve standalone EV observation order before refreshing SOH.
        c._observe(timestamp, float(status.get(f"{name} Electric Power (kW)", 0)), step)
        soh = float(status[f"{name} SOH (-)"])
        if not np.isfinite(soh) or not 0 < soh <= 1:
            raise ValueError("EV measured SOH must be in (0,1]")
        c.current_soh = soh
        c.current_capacity = c.ev.capacity_kwh * soh
        reported_capacity = status.get(f"{name} Capacity (kWh)", c.current_capacity)
        if not np.isclose(reported_capacity, c.current_capacity, rtol=1e-9, atol=1e-10):
            raise ValueError("EV measured SOH and capacity are inconsistent")
        if self.peak_arrival_comfort_enabled:
            c.seed_observed_peak_comfort(
                timestamp, forecast.ev_schedules[c.ev.vehicle_id], forecast,
                self._completed_ev_states,
            )
            c.observe_realized_peak_comfort(
                timestamp, forecast.ev_schedules[c.ev.vehicle_id], soc, forecast,
                parked=status.get(f"{name} Parked"),
            )
        c.soc_history.extend([float(status.get(f"{name} End SOC (-)", soc)), soc])
        c.soc_history = np.asarray(c.soc_history)[turning_points(c.soc_history)].tolist()
        return soc, step

    def _ev_control(self, timestamp: pd.Timestamp, soc: float, step: pd.Timedelta, forecast: Forecast):
        c = self.ev_controller
        mobility = forecast.ev_schedules[c.ev.vehicle_id]
        available = bool(mobility.available([timestamp])[0])
        fallback = c.fallback or c.next_solve is None
        if fallback:
            target = (min(c.ev.charger_power_kw,
                          max(0, c.comfort_target_at(timestamp) - soc) * c.current_capacity /
                          (step.total_seconds()/3600 * c.options.charge_efficiency))
                      if available else 0.0)
            mode, power, override = int(target > 0), target, False
        else:
            target = c.target
            # The plant's actual_backed_load callback independently clips V2H
            # against the realized HPWH and compressor-plus-fan draws.
            backed = max(-target, 0.0)
            mode, power, override = c._project(timestamp, target, soc, available, backed)
        backed = max(-target, 0.0)
        c.action_history.append(dict(timestamp=timestamp, requested_kw=target, projected_kw=power,
                                     mode=mode, planned_mode=getattr(c, "held_mode", None),
                                     one_shot=self.one_shot,
                                     projection_override=override, fallback=fallback))
        return {c.equipment_name: {"P Setpoint": power, "Backed Load (kW)": backed}}

    def _held_controls(self, timestamp, previous_status, soc, step, forecast):
        return {
            **self.water_heater_controller.held_controls_for_step(),
            **self.hvac_controller.held_controls_for_step(previous_status),
            **self._ev_control(timestamp, soc, step, forecast),
        }

    def _record_ev_solution(self, timestamp, soc, problem, plan, duration, successful, status):
        c = self.ev_controller
        c.fallback = not successful
        c.next_solve = timestamp + pd.Timedelta(minutes=c.options.control_minutes)
        c.solve_history.append(dict(timestamp=timestamp, status=status, successful=successful,
                                    measured_soh=c.current_soh, capacity_kwh=c.current_capacity,
                                    solve_seconds=duration, ocp_seconds=duration,
                                    limit_seconds=self.solver_options.get("TimeLimit"), fallback=c.fallback,
                                    scenario_count=len(plan["scenario_plans"]),
                                    mobility_scenario_count=plan["scenario_shape"][1],
                                    water_scenario_count=plan["scenario_shape"][0],
                                    scenario_ids=list(plan["scenario_ids"])))
        if not successful:
            return
        command = plan["shared_first_command"]
        if command is None:
            c.held_mode, c.target = 0, 0.0
        else:
            charge, discharge = (float(v.value) > 0.5 for v in command)
            c.held_mode = int(charge) - int(discharge)
            if discharge:
                first_rates = []
                for path in plan["scenario_plans"]:
                    horizon = path["horizon"]
                    active = np.flatnonzero((horizon.group == 0) & (horizon.z == 1))
                    if len(active):
                        first_rates.append(float(path["discharge"].value[int(active[0])]))
                if not first_rates or not np.isfinite(first_rates).all() or not np.allclose(
                    first_rates, first_rates[0], rtol=1e-4, atol=1e-4
                ):
                    raise ValueError("Joint EV paths disagree on the shared first discharge rate")
                c.target = -first_rates[0]
            else:
                c.target = c.ev.charger_power_kw if charge else 0.0
        first = plan["scenario_plans"][0]
        c.plans.append(dict(timestamp=timestamp, boundaries=first["boundaries"].astype(str).tolist(),
            soc=first["state"].value.tolist(), chemical_kw=first["chemical"].value.tolist(),
            charge_kw=first["charge"].value.tolist(), discharge_kw=first["discharge"].value.tolist(),
            objective=sum(float(v.value) for v in plan["objective_terms"].values()),
            objective_terms={k: float(v.value) for k, v in plan["objective_terms"].items()},
            degradation_method=c.options.degradation_method, measured_soh=c.current_soh,
            capacity_kwh=c.current_capacity, terminal_value_usd_per_kwh=first["terminal_value_usd_per_kwh"],
            **predicted_trip_energy_fields(first),
            initial_energy_kwh=c.current_capacity * soc,
            terminal_energy_kwh=c.current_capacity * float(first["state"].value[-1]),
            dispatched_target_kw=c.target, dispatched_mode=c.held_mode,
            normalization_steps=first["N"], scenario_metadata=plan["scenario_metadata"],
            representative_scenario_id=plan["scenario_ids"][0],
            scenarios=[dict(candidate_id=identifier, boundaries=p["boundaries"].astype(str).tolist(),
                            soc=p["state"].value.tolist(), chemical_kw=p["chemical"].value.tolist(),
                            charge_kw=p["charge"].value.tolist(), discharge_kw=p["discharge"].value.tolist(),
                            available=p["horizon"].z.tolist(), trip_energy_kwh=p["horizon"].trip_kwh.tolist(),
                            **predicted_trip_energy_fields(p),
                            departure_indices=p["horizon"].departure_indices.tolist(),
                            objective_terms={k: float(v.value) for k, v in p["objective_terms"].items()},
                            terminal_energy_kwh=p["capacity_kwh"]*float(p["state"].value[-1]))
                       for identifier, p in zip(plan["scenario_ids"], plan["scenario_plans"])]))

    def _record_plan(self, timestamp, water, hvac, rest_method, rest_diagnostics):
        water_power = np.asarray(water.electric_power_kw.value, dtype=float)
        hvac_power = np.asarray(hvac.electric_power_kw.value, dtype=float).reshape(-1)
        ev_power = np.asarray(self._ev_power_scenarios_kw.value, dtype=float)
        combined = np.asarray(self._modeled_home_power_scenarios_kw.value, dtype=float)
        index = pd.date_range(timestamp, periods=hvac_power.size, freq=self.hvac_controller.model_timestep)
        frame = pd.DataFrame(index=index)
        frame.index.name = "timestamp"
        frame["peak_objective_active"] = self._peak_objective_active
        frame["electricity_price_usd_per_kwh"] = np.asarray(water.prices, dtype=float)
        frame["hvac_electric_power_kw"] = hvac_power
        frame["hpwh_mean_electric_power_kw"] = water_power.mean(axis=0)
        frame["ev_mean_electric_power_kw"] = ev_power.mean(axis=0)
        frame["rest_of_home_forecast_kw"] = self._rest_of_home_forecast_kw
        if self._hub_other_forecast_kw is not None:
            frame["hub_other_forecast_kw"] = self._hub_other_forecast_kw
            frame["outside_hub_forecast_kw"] = self._outside_hub_forecast_kw
        frame["peak_limit_kw"] = self.peak_limit_kw if self.peak_limit_kw is not None else np.nan
        frame["modeled_home_mean_electric_power_kw"] = combined.mean(axis=0)
        frame["modeled_home_worst_scenario_power_kw"] = combined.max(axis=0)
        if self.uses_daily_grid2_demand:
            daily = self.daily_demand_metadata(solved=True)
            lookup = {day: i for i, day in enumerate(daily["daily_demand_days"])}
            row_days = [self._demand_local_time(t).normalize().isoformat() for t in index]
            frame["grid2_demand_formulation"] = self.grid2_demand_formulation
            frame["daily_demand_scale"] = daily["daily_demand_scale"]
            frame["daily_anytime_demand_active"] = daily["daily_demand_active"]
            for key in ("daily_on_peak_demand_kw", "daily_anytime_demand_kw",
                        "daily_demand_observed_on_peak_kw", "daily_demand_observed_anytime_kw"):
                values = daily[key]
                frame[key] = [np.nan if day not in lookup else values[lookup[day]] for day in row_days]
        frame["rest_of_home_forecast_method"] = rest_method
        for key, value in (rest_diagnostics or {}).items():
            frame[key] = str(value) if isinstance(value, dict) else value
        frame.attrs.update(solved_at=pd.Timestamp(timestamp), water_scenario_count=water_power.shape[0],
                           ev_scenario_count=combined.shape[0] // water_power.shape[0],
                           joint_scenario_count=combined.shape[0])
        self._optimization_plans.append(frame)

    def _write_decision_trace(self, timestamp: pd.Timestamp, record: Mapping[str, Any]) -> Path | None:
        """Persist one issue atomically, including inputs before the solver starts."""
        directory = self.decision_trace_directory
        if directory is None:
            return None
        issue_number = len(self._solve_history) + 1
        path = directory / f"joint_issue_{issue_number:04d}_{timestamp:%Y%m%d_%H%M%S}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(_trace_json_value(record), allow_nan=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
        return path

    @staticmethod
    def _replace_decision_trace(path: Path | None, record: Mapping[str, Any]) -> None:
        if path is None:
            return
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(_trace_json_value(record), allow_nan=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def _issued_decision_trace(
        self, timestamp: pd.Timestamp, forecast: Forecast, water, hvac,
        prediction: EVScenarioSet, ev_plan: Mapping[str, Any],
        hub_series: pd.Series | None, outside_series: pd.Series | None,
        rest_series: pd.Series | None, initial_soc: float,
    ) -> dict[str, Any]:
        """Capture the arrays actually supplied to this decision's MILP."""
        steps = int(self.horizon / self.hvac_controller.model_timestep)
        index = pd.date_range(timestamp, periods=steps, freq=self.hvac_controller.model_timestep)
        water_disturbances = np.asarray(water.disturbances, dtype=float)
        hvac_disturbances = np.asarray(hvac.disturbances, dtype=float)
        water_names = list(water.model["disturbance_names"])
        hvac_names = list(hvac.model["disturbance_names"])
        water_count = water_disturbances.shape[0]
        ev_count = len(prediction.forecasts)
        if (water_disturbances.shape != (water_count, len(water_names), steps)
                or hvac_disturbances.shape != (len(hvac_names), steps)
                or not self._valid_scenario_counts(water_count, ev_count)):
            raise ValueError("Joint trace inputs do not match the prepared optimization grids")
        raw_draws = getattr(water, "source_draws_l_per_min", None)
        if raw_draws is not None:
            raw_draws = np.asarray(raw_draws, dtype=float)
            if raw_draws.shape != (water_count, steps):
                raise ValueError("Selected water-draw paths do not match the optimization grid")
        water_selection = self.water_heater_controller._scenario_diagnostics_history[-1]
        if pd.Timestamp(water_selection["timestamp"]) != timestamp:
            raise ValueError("Water-scenario diagnostics do not match this decision")
        ev_ids = prediction.candidate_ids or tuple(str(i) for i in range(ev_count))
        ev_paths = []
        for ev_index in range(ev_count):
            horizon = ev_plan["scenario_plans"][ev_index]["horizon"]
            ev_paths.append({
                "scenario_index": ev_index,
                "candidate_id": ev_ids[ev_index],
                "probability": 1 / ev_count,
                "boundaries": horizon.t,
                "duration_hours": horizon.dt,
                "available": horizon.z,
                "trip_energy_kwh": horizon.trip_kwh,
                "control_interval_index": horizon.group,
                "departure_indices": horizon.departure_indices,
                "price_usd_per_kwh": horizon.price,
                "fixed_backed_tariff_cost_usd_per_hour": horizon.backed_cost,
                "reference_discharge_mean_kw": horizon.discharge_mean,
                "reference_discharge_min_kw": horizon.discharge_min,
            })
        fixed_backed_cost = float(np.mean([
            np.sum(path["horizon"].dt * path["horizon"].backed_cost) / path["horizon"].N
            for path in ev_plan["scenario_plans"][:ev_count]
        ])) * self.ev_controller.options.electricity_weight

        source_step = pd.Timedelta(forecast.data.index[1] - forecast.data.index[0])
        thermal_step = self.hvac_controller.model_timestep
        if thermal_step % source_step:
            raise ValueError("Forecast source grid does not divide the joint control step")
        samples_per_step = thermal_step // source_step
        weather_index = pd.date_range(
            timestamp, periods=steps * samples_per_step, freq=source_step
        )
        weather_rows = forecast.data.reindex(weather_index).copy()
        weather_run = None
        if forecast.weather_archive is not None:
            weather_run = forecast.weather_archive.selected_run_time(timestamp)
            issued_weather = forecast.weather_archive.weather_for_hvac(
                timestamp, weather_index, forecast.data.loc[timestamp], thermal_step,
            )
            weather_rows.loc[:, issued_weather.columns] = issued_weather
        weather_columns = (
            "Ambient Dry Bulb (C)", "Ambient Relative Humidity (-)",
            "GHI (W/m^2)", "DNI (W/m^2)", "DHI (W/m^2)",
            "zenith", "azimuth",
        )
        weather_means = {
            name: weather_rows[name].to_numpy(dtype=float).reshape(steps, samples_per_step).mean(axis=1)
            for name in weather_columns if name in weather_rows
        }
        return {
            "trace_version": 1,
            "perfect_forecasts": self.perfect_forecasts,
            "one_shot": self.one_shot,
            "rate_structure": self.rate_structure,
            "grid2_demand_formulation": self.grid2_demand_formulation,
            **self.daily_demand_metadata(solved=False),
            "normalization_steps": steps,
            "solver_time_limit_seconds": self.solver_options.get("TimeLimit"),
            "issue_time": timestamp,
            "status": "issued",
            "successful": None,
            "fallback": None,
            "objective_components": None,
            "objective_units": "weighted and horizon-normalized optimizer dollars; not the metered TOU bill",
            "target_starts": index,
            "peak_window_active": self._peak_objective_active,
            "peak_demand_interval_minutes": (
                30 if self.uses_daily_grid2_demand or self.peak_window == WINTER_TOU_PEAK_WINDOW else 15
            ),
            "peak_demand_block_starts": self._peak_demand_block_starts,
            "peak_observed_first_half_kw": self._peak_observed_first_half_kw,
            "peak_limit_kw": self.peak_limit_kw,
            "peak_weight_usd_per_kw": self.peak_weight_usd_per_kw,
            "peak_coefficient_usd_per_kw": (
                GRID2_ON_PEAK_DEMAND_USD_PER_KW_DAY * thermal_step.total_seconds() / 3600 / steps
                if self.uses_daily_grid2_demand else (
                    self.peak_weight_usd_per_kw * thermal_step.total_seconds() / 3600 / steps
                    if self.peak_limit_kw is not None else self.peak_weight_usd_per_kw
                )
            ),
            "initial_ev_soc": initial_soc,
            "initial_ev_capacity_kwh": self.ev_controller.current_capacity,
            "ev_terminal_value_provenance": ev_plan["scenario_plans"][0].get("terminal_value_provenance"),
            "solver_backend": self.solver_backend,
            "weather_run_utc": weather_run,
            "weather_15min_means": weather_means,
            "hvac": {
                "initial_state": hvac.initial_state,
                "disturbances_by_name": dict(zip(hvac_names, hvac_disturbances)),
                "price_usd_per_kwh": hvac.prices,
                "cop": hvac.cop,
                "scheduled_setpoint_c": hvac.setpoints,
                "endpoint_comfort_lower_c": hvac.lower,
                "endpoint_comfort_upper_c": hvac.upper,
                "historical_degree_hours": hvac.historical_degree_hours,
            },
            "water_heater": {
                "initial_state": water.initial_state,
                "price_usd_per_kwh": water.prices,
                "cop": (water.model["rc"].cop
                        if self.water_heater_controller.model_kind == "rc"
                        else self.water_heater_controller.cop),
                "scenario_probability": 1 / water_count,
                "selection_diagnostics": water_selection,
                "raw_selected_draw_l_per_min": raw_draws,
                "effective_disturbances_by_scenario": [
                    dict(zip(water_names, water_disturbances[scenario]))
                    for scenario in range(water_count)
                ],
            },
            "ev_mobility_scenarios": ev_paths,
            "joint_scenario_ids": ev_plan["scenario_ids"],
            "joint_scenario_probability": 1 / (water_count * ev_count),
            "ev_fixed_backed_tariff_cost_usd": fixed_backed_cost,
            "ev_fixed_backed_cost_role": (
                "Retrospective non-V2H backed-load tariff constant; additive and "
                "independent of every current joint decision. EV variable electricity "
                "cost is logged separately after a successful solve."
            ),
            "hub_other_forecast_kw": None if hub_series is None else hub_series.to_numpy(dtype=float),
            "hub_other_forecast_diagnostics": None if hub_series is None else dict(hub_series.attrs),
            "outside_hub_forecast_kw": None if outside_series is None else outside_series.to_numpy(dtype=float),
            "outside_hub_forecast_diagnostics": None if outside_series is None else dict(outside_series.attrs),
            "aggregate_rest_forecast_kw": None if rest_series is None else rest_series.to_numpy(dtype=float),
            "aggregate_rest_forecast_diagnostics": None if rest_series is None else dict(rest_series.attrs),
            "outside_hub_target_definition": (
                "outside Hub excludes electric resistance; explicit HVAC model includes resistance"
                if hub_series is not None else "aggregate rest-of-home forecast"
            ),
        }

    def _refresh_gurobi_environment(self, problem: cp.Problem | None = None) -> None:
        """Renew the WLS session without changing the compiled optimization model.

        The long replay can outlive a single default Gurobi environment. CVXPY
        retains the last native model on a Problem, so release that reference
        before disposing of the default environment after a failed solve.
        """
        if self._fast_solver is not None:
            self._fast_solver.close()
        if problem is not None:
            problem._solver_cache.clear()
            problem._solver_stats = None
        gc.collect()
        gp.disposeDefaultEnv()
        self._gurobi_environment_refreshed_at = time.perf_counter()
        self._gurobi_environment_refresh_count += 1

    def _solve_problem_with_retries(self, problem: cp.Problem) -> None:
        """Retry transient WLS acquisition without advancing simulated time."""
        self._solver_retry_count = 0
        self._wls_network_retry_count = 0
        if time.perf_counter() - self._gurobi_environment_refreshed_at >= WLS_REFRESH_SECONDS:
            self._refresh_gurobi_environment()
        while True:
            try:
                self._fast_solver_metrics = {}
                if self.solver_backend == "sparse_native_v1":
                    if self._fast_solver is None:
                        self._fast_solver = ReusableJointMILPSolver()
                    try:
                        self._fast_solver.solve(problem, dict(self.solver_options))
                        self._fast_solver_metrics = dict(self._fast_solver.last_metrics)
                    except UnsupportedFastMILP as error:
                        self._fast_solver.close()
                        problem.solve(solver="GUROBI", **dict(self.solver_options))
                        self._fast_solver_metrics = {"backend": "cvxpy", "fallback_reason": str(error)}
                else:
                    problem.solve(solver="GUROBI", **dict(self.solver_options))
                return
            except gp.GurobiError as error:
                if (not _transient_wls_network_error(error)
                        or self._wls_network_retry_count == len(WLS_DNS_RETRY_DELAYS_SECONDS)):
                    raise
                self._refresh_gurobi_environment(problem)
                delay = WLS_DNS_RETRY_DELAYS_SECONDS[self._wls_network_retry_count]
                # Adjacent parallel runs should not reacquire a WLS token
                # in exactly the same instant after a short DNS outage.
                time.sleep(delay + (os.getpid() % 7) * 0.19)
                self._wls_network_retry_count += 1
            except cp.error.SolverError:
                if self._solver_retry_count == 2:
                    raise
                self._refresh_gurobi_environment(problem)
                self._solver_retry_count += 1

    def _freeze_one_shot_plan(self, timestamp, water, hvac, ev_plan) -> None:
        """Copy the sole accepted primal schedule before releasing its model."""
        if ev_plan["scenario_shape"] != (1, 1):
            raise ValueError("One-shot whole-home replay requires one exact water/EV path")
        path = ev_plan["scenario_plans"][0]
        starts = pd.date_range(timestamp, periods=int(self.horizon / self.hvac_controller.model_timestep),
                               freq=self.hvac_controller.model_timestep)
        boundaries = pd.DatetimeIndex(path["boundaries"])
        modes = (np.asarray(path["charge_mode"].value > .5, dtype=int)
                 - np.asarray(path["discharge_mode"].value > .5, dtype=int))
        if boundaries[0] != timestamp or boundaries[-1] != timestamp + self.horizon:
            raise ValueError("One-shot EV schedule does not span the requested full period")
        plan = {
            "issued_at": pd.Timestamp(timestamp), "end": timestamp + self.horizon,
            "rate_structure": self.rate_structure,
            "grid2_demand_formulation": self.grid2_demand_formulation,
            **self.daily_demand_metadata(solved=True),
            "normalization_steps": len(starts),
            "thermal_starts": starts,
            "hvac_heat_w": np.asarray(hvac.heat_hvac.value, dtype=float).reshape(-1).copy(),
            "hvac_heat_pump_heat_w": np.asarray(hvac.heat_hp.value, dtype=float).reshape(-1).copy(),
            "hvac_resistance_heat_w": np.asarray(hvac.heat_er.value, dtype=float).reshape(-1).copy(),
            "hvac_training_setpoint_c": np.asarray(hvac.setpoints, dtype=float).reshape(-1).copy(),
            "hvac_predicted_temperature_c": _predicted_hvac_temperature_c(hvac),
            "hvac_mixed_integer_states": {
                key: np.asarray(expression.value).copy()
                for key, expression in getattr(hvac, "mixed_int_state_expressions", {}).items()
            },
            "hpwh_heat_kw": np.asarray(water.heat.value, dtype=float)[0].copy(),
            "hpwh_on_mode": np.asarray(water.heat.value, dtype=float)[0] > .1,
            "ev_boundaries": boundaries, "ev_modes": modes.copy(),
            "ev_charge_mode": np.asarray(path["charge_mode"].value > .5, dtype=int),
            "ev_discharge_mode": np.asarray(path["discharge_mode"].value > .5, dtype=int),
            "ev_charge_kw": np.asarray(path["charge"].value, dtype=float).reshape(-1).copy(),
            "ev_discharge_kw": np.asarray(path["discharge"].value, dtype=float).reshape(-1).copy(),
            "ev_soc": np.asarray(path["state"].value, dtype=float).reshape(-1).copy(),
            "ev_available": np.asarray(path["horizon"].z, dtype=int).copy(),
            "ev_trip_energy_kwh": np.asarray(path["horizon"].trip_kwh, dtype=float).copy(),
            **{f"ev_{key}": value for key, value in predicted_trip_energy_fields(path).items()},
            "replay_semantics": (
                "One accepted joint MILP; fixed thermal schedule translated through the "
                "existing thermostat command mapping, and fixed EV modes/solved discharge "
                "rates projected against measured SOC, availability and actual backed load. "
                "Plant observation continues; no future problem is prepared or solved."
            ),
        }
        for key in ("hvac_heat_w", "hvac_heat_pump_heat_w", "hvac_resistance_heat_w",
                    "hvac_training_setpoint_c", "hpwh_heat_kw"):
            if plan[key].shape != (len(starts),) or not np.isfinite(plan[key]).all():
                raise ValueError(f"One-shot accepted plan has an invalid {key} trajectory")
        if (plan["hvac_predicted_temperature_c"].shape != (len(starts) + 1,)
                or len(boundaries) != len(modes) + 1
                or plan["ev_discharge_kw"].shape != modes.shape
                or not np.isfinite(plan["ev_discharge_kw"]).all()):
            raise ValueError("One-shot accepted plan has incompatible boundary trajectories")
        self._one_shot_plan = plan

    def _observe_one_shot_thermal_step(self, timestamp, previous_status, forecast) -> None:
        """Keep measured histories current without constructing another MILP."""
        if previous_status is None:
            raise ValueError("One-shot whole-home replay requires measured thermal states")
        water = self.water_heater_controller
        water._observe_heat_pump(timestamp, previous_status)
        nominal = previous_status.get(f"{water.equipment_name} Nominal Heat Pump Capacity (W)")
        if nominal is not None:
            water.nominal_hp_capacity_kw = float(nominal) / 1000
        model = water._load_model()
        if model is None:
            raise ValueError("One-shot water-heater model is unavailable during replay")
        measurement = water._measurement(previous_status, model["output_names"])
        if measurement is None:
            raise ValueError("One-shot water-heater measurements are unavailable during replay")
        water._observe_legionella_temperature(timestamp, measurement)
        # This observation path is also used for thermal-controller warm-up;
        # it updates measurements/model memory without preparing future inputs.
        self.hvac_controller.record_warmup_step(timestamp, previous_status, forecast)

    def _replay_one_shot_controls(self, timestamp, previous_status, soc, step, forecast,
                                 *, observe=True, thermal_controls=None):
        plan = self._one_shot_plan
        if plan is None:
            raise RuntimeError("One-shot whole-home control has no accepted schedule to replay")
        if timestamp < plan["issued_at"] or timestamp >= plan["end"]:
            raise ValueError("One-shot whole-home schedule does not cover the requested plant minute")
        if observe:
            self._observe_one_shot_thermal_step(timestamp, previous_status, forecast)
        thermal_index = int(plan["thermal_starts"].searchsorted(timestamp, side="right")) - 1
        ev_index = int(plan["ev_boundaries"].searchsorted(timestamp, side="right")) - 1
        if thermal_controls is None and thermal_index != self._one_shot_thermal_index:
            water = self.water_heater_controller
            water._last_setpoint = (water.maximum_setpoint_c if plan["hpwh_heat_kw"][thermal_index] > .1
                                    else water.minimum_setpoint_c)
            water._last_decision_time = timestamp
            self.hvac_controller._start_setpoint_interval(
                previous_status,
                planned_heat_w=float(plan["hvac_heat_w"][thermal_index]),
                planned_hp_heat_w=float(plan["hvac_heat_pump_heat_w"][thermal_index]),
                planned_er_heat_w=float(plan["hvac_resistance_heat_w"][thermal_index]),
                training_setpoint_c=float(plan["hvac_training_setpoint_c"][thermal_index]),
                predicted_next_temperature_c=float(plan["hvac_predicted_temperature_c"][thermal_index + 1]),
            )
            self.hvac_controller._last_decision_time = timestamp
        self._one_shot_thermal_index = thermal_index
        ev = self.ev_controller
        ev.held_mode = int(plan["ev_modes"][ev_index])
        ev.target = (ev.ev.charger_power_kw if ev.held_mode == 1 else
                     -float(plan["ev_discharge_kw"][ev_index]) if ev.held_mode == -1 else 0.0)
        if thermal_controls is None:
            thermal_controls = {
                **self.water_heater_controller.held_controls_for_step(),
                **self.hvac_controller.held_controls_for_step(previous_status),
            }
        controls = {**thermal_controls, **self._ev_control(timestamp, soc, step, forecast)}
        self._one_shot_replay_history.append({
            "timestamp": timestamp, "issued_at": plan["issued_at"],
            "thermal_interval_index": thermal_index, "ev_block_index": ev_index,
            "ev_planned_mode": ev.held_mode, "ev_planned_target_kw": ev.target,
            "controls_json": json.dumps(_trace_json_value(controls), allow_nan=False),
        })
        return controls

    def controls_for_step(
        self, timestamp: pd.Timestamp, previous_status: Mapping[str, Any] | None,
        current_schedule: Mapping[str, Any], forecast: Forecast,
    ) -> dict[str, dict[str, Any]]:
        timestamp = pd.Timestamp(timestamp)
        soc, step = self._observe_ev(timestamp, previous_status, forecast)
        if self.one_shot and self._one_shot_attempted:
            return self._replay_one_shot_controls(timestamp, previous_status, soc, step, forecast)
        water = self.water_heater_controller.prepare_milp_scenario_step(
            timestamp, previous_status, current_schedule, forecast
        )
        hvac = self.hvac_controller.prepare_mixed_int_step(
            timestamp, previous_status, current_schedule, forecast
        )
        if water is None or hvac is None:
            if self.one_shot:
                raise RuntimeError("One-shot whole-home control could not prepare its initial full-period problem")
            return self._held_controls(timestamp, previous_status, soc, step, forecast)
        if getattr(self.ev_controller, "terminal_energy_value_method", "daily_mean") != "daily_mean":
            if self.rate_structure != MISO_RTP_DOE:
                raise ValueError("Historical terminal energy value is approved only for Grid 3")
            self.ev_controller.set_terminal_availability_history(
                forecast.ev_schedules[self.ev_controller.ev.vehicle_id]
            )
        prediction = forecast.for_device(self.ev_controller.equipment_name, timestamp, self.ev_controller.horizon)
        if not isinstance(prediction, EVScenarioSet):
            raise ValueError("Whole-home control requires an EV scenario forecast")
        ev_count = len(prediction.forecasts)
        water_count = water.electric_power_kw.shape[0]
        if not self._valid_scenario_counts(water_count, ev_count):
            raise ValueError("Whole-home control requires five or seven EV scenarios")
        steps = int(self.horizon / self.hvac_controller.model_timestep)
        split = (getattr(forecast, "hub_other_forecast", None) is not None
                 or getattr(forecast, "outside_hub_forecast", None) is not None)
        if split:
            if forecast.hub_other_forecast is None or forecast.outside_hub_forecast is None:
                raise ValueError("Joint split forecasts require both Hub-other and Outside-Hub")
            hub_provider = forecast.hub_other_forecast
            outside_provider = forecast.outside_hub_forecast
            hub_series = hub_provider.forecast_kw(timestamp, steps=steps)
            outside_series = outside_provider.forecast_kw(timestamp, steps=steps)
            rest_series = None
        else:
            rest_provider = forecast.rest_of_home_forecast
            if rest_provider is None:
                raise ValueError("Whole-home peak control requires background forecasts")
            rest_series = rest_provider.forecast_kw(timestamp, steps=steps)
            hub_series = outside_series = None
        # The HPWH scenarios, HVAC compressor/fan decisions, and fixed
        # Hub-other forecast set eligible discharge. Resistance and Outside-Hub
        # remain in whole-home power but cannot raise the discharge cap.
        hvac_hp_fan_power_kw = hvac.electric_power_kw - hvac.heat_er / 1000
        ev_problem, ev_plan = build_joint_ev_scenario_problem(
            self.ev_controller, timestamp, soc, prediction,
            water.electric_power_kw, hvac_hp_fan_power_kw,
            self.hvac_controller.model_timestep,
            eligible_power_upper_kw=self._eligible_power_upper_kw(
                water, hvac, None if hub_series is None else hub_series.to_numpy(dtype=float),
            ),
            hub_other_kw=None if hub_series is None else hub_series.to_numpy(dtype=float),
            perfect_forecasts=self.perfect_forecasts,
        )
        observed_first_half_kw = None
        if (self.peak_window == WINTER_TOU_PEAK_WINDOW
                and self.peak_weight_usd_per_kw > 0
                and timestamp.minute % 30 == 15
                and WinterTOURate.is_on_peak(timestamp - pd.Timedelta(minutes=15))) or (
                    self.uses_daily_grid2_demand and timestamp.minute % 30 == 15
                    and self._demand_local_time(timestamp).weekday() < 5):
            observed_first_half_kw = self._observed_first_half_kw(timestamp)
        problem = self.optimization_problem_milp(
            water, hvac, ev_problem, ev_plan,
            None if rest_series is None else rest_series.to_numpy(dtype=float),
            hub_other_kw=None if hub_series is None else hub_series.to_numpy(dtype=float),
            outside_hub_kw=None if outside_series is None else outside_series.to_numpy(dtype=float),
            observed_first_half_kw=observed_first_half_kw,
        )
        decision_trace = (
            self._issued_decision_trace(
                timestamp, forecast, water, hvac, prediction, ev_plan,
                hub_series, outside_series, rest_series, soc,
            )
            if self.decision_trace_directory is not None else None
        )
        trace_path = (
            self._write_decision_trace(timestamp, decision_trace)
            if decision_trace is not None else None
        )
        self._notify_progress(dict(timestamp=timestamp, status="solving",
                                   water_scenario_count=water_count, ev_scenario_count=ev_count,
                                   joint_scenario_count=water_count * ev_count))
        self._one_shot_attempted = self.one_shot
        started = time.perf_counter()
        try:
            self._solve_problem_with_retries(problem)
        except (cp.error.SolverError, gp.GurobiError) as error:
            duration = time.perf_counter() - started
            self.solve_durations_s["whole_home_joint_milp"].append(duration)
            self._solve_history.append(dict(timestamp=timestamp, optimization_kind="whole_home_joint_milp",
                water_scenario_count=water_count, ev_scenario_count=ev_count, joint_scenario_count=water_count * ev_count,
                duration_s=duration, status="solver_error", successful=False, solver_error=str(error),
                solver_retry_count=self._solver_retry_count,
                wls_network_retry_count=self._wls_network_retry_count,
                gurobi_environment_refresh_count=self._gurobi_environment_refresh_count))
            if decision_trace is not None:
                decision_trace.update(
                    status="solver_error", successful=False, fallback=None,
                    solver=self._solve_history[-1],
                )
                self._replace_decision_trace(trace_path, decision_trace)
            self._record_ev_solution(timestamp, soc, problem, ev_plan, duration, False, "solver_error")
            self._notify_progress(self._solve_history[-1])
            raise RuntimeError(f"Whole-home Gurobi solve failed at {timestamp}: {error}") from error
        duration = time.perf_counter() - started
        status = str(problem.status)
        values_ready = (water.heat.value is not None and all(
            expression.value is not None for expression in (
                hvac.heat_hvac, hvac.heat_hp, hvac.heat_er,
                *(p["state"] for p in ev_plan["scenario_plans"]),
            )
        ))
        feasible = values_ready and self.hvac_controller._time_limited_solution_is_feasible(problem)
        successful = status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE, cp.USER_LIMIT} and feasible
        metrics = self.hvac_controller._solver_metrics(problem)
        metrics["solver_backend"] = self.solver_backend
        metrics["fast_solver_metrics"] = dict(self._fast_solver_metrics)
        row = dict(timestamp=timestamp, optimization_kind="whole_home_joint_milp",
                   water_scenario_count=water_count, ev_scenario_count=ev_count, joint_scenario_count=water_count * ev_count,
                   duration_s=duration, status=status, successful=successful,
                   candidate_values_ready=values_ready, candidate_feasible=feasible,
                   solver_retry_count=self._solver_retry_count,
                   wls_network_retry_count=self._wls_network_retry_count,
                   gurobi_environment_refresh_count=self._gurobi_environment_refresh_count, **metrics)
        self.solve_durations_s["whole_home_joint_milp"].append(duration)
        self._solve_history.append(row)
        if decision_trace is not None:
            decision_trace.update(
                status=status, successful=successful, fallback=not successful, solver=row,
            )
            if not successful:
                decision_trace["primal_diagnostics"] = dict(
                    triggers=[dict(kind="rejected_candidate", values_ready=values_ready,
                                   feasible=feasible, status=status)],
                    candidate=collect_primal_diagnostics(problem),
                )
            self._replace_decision_trace(trace_path, decision_trace)
        self._record_ev_solution(timestamp, soc, problem, ev_plan, duration, successful, status)
        self._notify_progress(row)
        if not successful:
            if self.one_shot:
                raise RuntimeError(f"One-shot whole-home optimization has no accepted full-period schedule at {timestamp}: {status}")
            if status == cp.USER_LIMIT and not self.peak_arrival_comfort_enabled:
                raise RuntimeError(f"Whole-home solve hit its time limit without a verified feasible solution at {timestamp}")
            return self._held_controls(timestamp, previous_status, soc, step, forecast)
        values = {key: float(np.asarray(expression.value).squeeze())
                  for key, expression in self._objective_components.items()}
        objective = float(problem.value)
        if decision_trace is not None:
            numerical_flags = _candidate_numerical_flags(objective, values, ev_plan["scenario_plans"])
            decision_trace["numerical_audit_flags"] = numerical_flags
            if numerical_flags:
                decision_trace["primal_diagnostics"] = dict(
                    triggers=numerical_flags, candidate=collect_primal_diagnostics(problem),
                )
        daily_components = {}
        if self.uses_daily_grid2_demand:
            daily_components = {
                "daily_demand_on_peak_component": float(self._daily_demand_objective_terms["on_peak"].value),
                "daily_demand_anytime_component": float(self._daily_demand_objective_terms["anytime"].value),
                "grid2_demand_formulation": self.grid2_demand_formulation,
                "daily_demand_scale": self._daily_demand_plan["daily_demand_scale"],
                "daily_demand_metadata_json": json.dumps(self.daily_demand_metadata(solved=True), allow_nan=False),
            }
        self._objective_component_history.append(dict(timestamp=timestamp, **values,
            **daily_components,
            scalar_objective_cost=objective, objective_balance_error=objective - sum(values.values())))
        if decision_trace is not None:
            water_power = np.asarray(water.electric_power_kw.value, dtype=float)
            hvac_power = np.asarray(hvac.electric_power_kw.value, dtype=float).reshape(-1)
            hvac_er = np.asarray(hvac.heat_er.value, dtype=float).reshape(-1) / 1000
            ev_power = np.asarray(self._ev_power_scenarios_kw.value, dtype=float)
            modeled_home = np.asarray(self._modeled_home_power_scenarios_kw.value, dtype=float)
            eligible = water_power + (hvac_power - hvac_er)[None, :]
            if hub_series is not None:
                eligible += hub_series.to_numpy(dtype=float)[None, :]
            decision_trace.update(
                objective_components=values,
                **self.daily_demand_metadata(solved=True),
                scalar_objective_cost=objective,
                objective_balance_error=objective - sum(values.values()),
                ev_variable_electricity_cost_usd=(
                    values["ev_electricity_objective_cost"]
                    - decision_trace["ev_fixed_backed_tariff_cost_usd"]
                ),
                solved_prediction={
                    "endpoint_times": pd.date_range(
                        timestamp, periods=steps + 1,
                        freq=self.hvac_controller.model_timestep,
                    ),
                    "hvac_indoor_temperature_c": _predicted_hvac_temperature_c(hvac),
                    "hvac_electric_power_kw": hvac_power,
                    "hvac_heat_pump_plus_fan_kw": hvac_power - hvac_er,
                    "hvac_electric_resistance_kw": hvac_er,
                    "water_heater_thermal_outputs_by_scenario_c": _predicted_water_outputs_c(
                        water, self.water_heater_controller.model_kind,
                        self.water_heater_controller.model_timestep,
                    ),
                    "water_heater_power_by_scenario_kw": water_power,
                    "eligible_backed_power_by_water_scenario_kw": eligible,
                    "ev_power_by_joint_scenario_kw": ev_power,
                    "modeled_home_power_by_joint_scenario_kw": modeled_home,
                },
                ev_objective_terms_by_joint_scenario=[
                    {
                        "candidate_id": identifier,
                        **predicted_trip_energy_fields(path),
                        "objective_terms": {
                            name: float(expression.value)
                            for name, expression in path["objective_terms"].items()
                        },
                    }
                    for identifier, path in zip(
                        ev_plan["scenario_ids"], ev_plan["scenario_plans"]
                    )
                ],
            )
            self._replace_decision_trace(trace_path, decision_trace)
        if split:
            self._record_plan(
                timestamp, water, hvac,
                f"{hub_provider.method_label} + {outside_provider.method_label}",
                {"hub_other": dict(hub_series.attrs), "outside_hub": dict(outside_series.attrs)},
            )
        else:
            self._record_plan(timestamp, water, hvac, rest_provider.method_label, rest_series.attrs)
        if self.one_shot:
            self._freeze_one_shot_plan(timestamp, water, hvac, ev_plan)
            if decision_trace is not None:
                decision_trace["accepted_one_shot_plan"] = self._one_shot_plan
                self._replace_decision_trace(trace_path, decision_trace)
        thermal_controls = {
            **self.water_heater_controller.apply_milp_scenario_solution(water),
            **self.hvac_controller.apply_mixed_int_solution(hvac),
        }
        if self.one_shot:
            return self._replay_one_shot_controls(
                timestamp, previous_status, soc, step, forecast,
                observe=False, thermal_controls=thermal_controls,
            )
        return {**thermal_controls, **self._ev_control(timestamp, soc, step, forecast)}
