"""EV-only scenario MPC with issued Home Hub and outside-Hub forecasts.

The EV mobility paths retain the established EV objective and shared first
mode. A separate whole-home peak epigraph uses the two load forecasts. Only
the EV is controlled; OCHRE supplies the minute-level physical V2H cap.
"""

from __future__ import annotations

from dataclasses import replace
import json
import time
from typing import Any, Callable, Mapping, Protocol

import cvxpy as cp
import numpy as np
import pandas as pd

from examples.five_home_block.five_home_block.ev_degradation import turning_points

from ..control_forecasts import EVScenarioSet
from ..ev_hub_forecast import EVHubIntervalForecast, INTERVAL, MOBILITY_SPLIT_POLICIES
from ..ev_home_hub_trace import predicted_trip_energy_fields
from ..forecast import Forecast, PRICE_COLUMN
from ..home_hub_observation import (
    TOTAL_POWER_KEY, observation_keys, validate_ev_only_hub_observation,
)
from ..rates import (
    DEMAND_CHARGE_USD_PER_KW_DAY, LOCAL_TIMEZONE, MISO_RTP_DOE,
    GRID2_ON_PEAK_DEMAND_USD_PER_KW_DAY, GRID2_ANYTIME_DEMAND_USD_PER_KW_DAY,
    WINTER_TOU, WINTER_TOU_DEMAND, WINTER_TOU_DEMAND_CPP, WinterTOURate,
)
from ..rest_of_home_forecast import RestOfHomeHybridForecast
from .ev import EVScenarioController
from .fast_joint_milp import ReusableJointMILPSolver, UnsupportedFastMILP
from .solver import solve_ocp


class HomeHubForecastProvider(Protocol):
    """Model-specific provider; the controller never reads future plant load."""

    def issue(
        self, timestamp: pd.Timestamp, completed_history: pd.DataFrame,
        horizon: pd.Timedelta, inverter_kw: float,
    ) -> EVHubIntervalForecast:
        """Return one forecast using observations completed by ``timestamp``."""
        ...


def _mean_ev_power_on_intervals(
    plan: Mapping[str, Any], starts: pd.DatetimeIndex, *, power_key: str = "ac",
) -> cp.Expression:
    """Integrate signed EV AC power over exact 15-minute peak intervals."""
    boundaries = pd.DatetimeIndex(plan["boundaries"])
    power = plan[power_key]
    if len(boundaries) != power.size + 1:
        raise ValueError("EV path power and mobility boundaries have incompatible lengths")
    # Vectorized overlap also keeps full-week oracle construction inexpensive.
    overlap = np.maximum(0, np.minimum(
        (starts + INTERVAL).asi8[:, None], boundaries.asi8[None, 1:],
    ) - np.maximum(starts.asi8[:, None], boundaries.asi8[None, :-1]))
    if not np.all(overlap.sum(axis=1) == INTERVAL.value):
        raise ValueError("EV path does not cover a complete 15-minute peak interval")
    weights = overlap / INTERVAL.value
    return weights @ power


class EVHomeHubController(EVScenarioController):
    """Control the EV alone against causal load forecasts and a live OCHRE cap."""

    def __init__(
        self, ev, hub_forecast_provider: HomeHubForecastProvider,
        outside_hub_forecast: RestOfHomeHybridForecast,
        *, peak_weight_usd_per_kw: float = 0.1,
        peak_limit_kw: float | None = None,
        peak_window: str = "night",
        scenario_count: int = 7,
        rate_structure: str = "legacy",
        one_shot: bool = False,
        grid2_demand_formulation: str = "weekday_daily_dt_over_n_v1",
        solver_backend: str = "cvxpy",
        hub_mobility_split_policy: str = "reject",
        peak_arrival_comfort_enabled: bool = False,
        hard_panel_limit_kw: float | None = None,
        progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    ):
        super().__init__(ev)
        if self.options.formulation != "milp" or not self.options.v2h:
            raise ValueError("EV Home Hub control requires a bidirectional EV MILP")
        if solver_backend not in {"cvxpy", "sparse_native_v1"}:
            raise ValueError("Unknown EV Home Hub solver backend")
        if solver_backend != "cvxpy" and self.options.solver != "GUROBI":
            raise ValueError("The sparse native EV backend requires Gurobi")
        self.solver_backend = solver_backend
        if hub_mobility_split_policy not in MOBILITY_SPLIT_POLICIES:
            raise ValueError("Unknown Home Hub mobility split policy")
        self.hub_mobility_split_policy = hub_mobility_split_policy
        self._fast_solver: ReusableJointMILPSolver | None = None
        self._solver_problem: cp.Problem | None = None
        self._solver_history_length = 0
        if self.options.control_minutes != 15 or (
            not one_shot and self.horizon != pd.Timedelta(hours=8)
            and not (peak_arrival_comfort_enabled and self.horizon == pd.Timedelta(hours=20))
        ):
            raise ValueError("EV Home Hub control requires 15-minute decisions and an eight-hour horizon, "
                             "or the selected 20-hour observed-peak-comfort variant")
        if self.horizon <= pd.Timedelta(0) or self.horizon % INTERVAL:
            raise ValueError("EV Home Hub horizon must contain positive complete 15-minute intervals")
        if rate_structure not in {"legacy", WINTER_TOU, WINTER_TOU_DEMAND, WINTER_TOU_DEMAND_CPP, MISO_RTP_DOE}:
            raise ValueError(f"Unknown EV Home Hub rate structure {rate_structure!r}")
        if hard_panel_limit_kw is not None:
            if (rate_structure not in {WINTER_TOU, WINTER_TOU_DEMAND, WINTER_TOU_DEMAND_CPP}
                    or isinstance(hard_panel_limit_kw, bool)
                    or not np.isfinite(hard_panel_limit_kw) or hard_panel_limit_kw <= 0):
                raise ValueError("Hard EV-only panel cap needs a positive Grid 1 or 2 limit")
        if one_shot and scenario_count != 1:
            raise ValueError("One-shot EV control requires one perfect mobility path")
        if not np.isfinite(peak_weight_usd_per_kw) or peak_weight_usd_per_kw < 0 or (
            rate_structure == "legacy" and peak_weight_usd_per_kw == 0
        ):
            raise ValueError("EV Home Hub peak weight must be positive and finite")
        if peak_limit_kw is not None and (not np.isfinite(peak_limit_kw) or peak_limit_kw <= 0):
            raise ValueError("EV Home Hub peak limit must be positive and finite")
        if peak_window not in {"all", "evening", "night"}:
            raise ValueError("EV Home Hub peak window must be all, evening, or night")
        if isinstance(scenario_count, bool) or not isinstance(scenario_count, int) or scenario_count < 1:
            raise ValueError("EV Home Hub scenario count must be a positive integer")
        self.hub_forecast_provider = hub_forecast_provider
        self.outside_hub_forecast = outside_hub_forecast
        self.include_total_home_meter = rate_structure != "legacy" or bool(
            getattr(outside_hub_forecast, "requires_total_home_meter", False)
        )
        self.rate_structure = rate_structure
        self.hard_panel_limit_kw = (None if hard_panel_limit_kw is None
                                    else float(hard_panel_limit_kw))
        if peak_arrival_comfort_enabled:
            self.configure_observed_peak_comfort(rate_structure)
        if grid2_demand_formulation not in {"weekday_daily_dt_over_n_v1", "legacy_horizon_peak"}:
            raise ValueError("Unknown Grid 2 demand formulation")
        self.grid2_demand_formulation = grid2_demand_formulation
        self.one_shot = bool(one_shot)
        self._one_shot_attempted = False
        self._one_shot_boundaries: pd.DatetimeIndex | None = None
        self._one_shot_modes: np.ndarray | None = None
        demand_rate = rate_structure in {WINTER_TOU_DEMAND, WINTER_TOU_DEMAND_CPP}
        self.peak_weight_usd_per_kw = (
            float(peak_weight_usd_per_kw) if rate_structure == "legacy"
            else GRID2_ON_PEAK_DEMAND_USD_PER_KW_DAY if self.uses_daily_grid2_demand
            else DEMAND_CHARGE_USD_PER_KW_DAY if demand_rate else 0.0
        )
        self.peak_limit_kw = (None if rate_structure != "legacy" or peak_limit_kw is None
                              else float(peak_limit_kw))
        self.peak_window = "winter_tou" if demand_rate else peak_window
        self.scenario_count = scenario_count
        self.progress_callback = progress_callback
        self._completed_plant_rows: list[tuple[pd.Timestamp, dict[str, Any]]] = []
        # Diagnostic reconstruction restores complete earlier-today maxima here;
        # live control obtains them from the completed aggregate-meter history.
        self._restored_daily_demand_peaks: dict[str, dict[str, float]] = {}
        self._issued_hub: EVHubIntervalForecast | None = None
        self.issued_load_history: list[dict[str, Any]] = []
        # These traces are passive diagnostics. They contain the exact issued
        # inputs and accepted scenario predictions, never future plant data.
        self.forecast_trace: list[pd.DataFrame] = []
        self.ev_scenario_input_trace: list[pd.DataFrame] = []
        self.scenario_prediction_trace: list[pd.DataFrame] = []
        self.trace_callback: Callable[[str, pd.DataFrame], None] | None = None
        self._last_built_plan: dict[str, Any] | None = None

    def _winter_demand_blocks(self, power, starts):
        """The agreed single horizon maximum uses complete 30-minute blocks."""
        active = np.zeros(len(starts), dtype=bool)
        blocks, block_starts = [], []
        observed_first_half = None
        if starts[0].minute % 30 == 15:
            block_start = starts[0] - INTERVAL
            if WinterTOURate.is_on_peak(block_start):
                expected = pd.date_range(block_start + pd.Timedelta(minutes=1), starts[0], freq="min")
                observed = [(time, row[TOTAL_POWER_KEY]) for time, row in self._completed_plant_rows
                            if block_start < time <= starts[0]]
                if [time for time, _ in observed] != list(expected):
                    raise ValueError("Winter demand control needs 15 completed whole-home meter minutes before a mid-block decision")
                observed_first_half = float(np.mean([value for _, value in observed]))
                if not np.isfinite(observed_first_half):
                    raise ValueError("Completed whole-home meter power must be finite")
                blocks.append((power[:, 0] + observed_first_half) / 2)
                block_starts.append(block_start)
                active[0] = True
        for index, start in enumerate(starts[:-1]):
            if start.minute % 30 == 0 and WinterTOURate.is_on_peak(start) and WinterTOURate.is_on_peak(starts[index + 1]):
                blocks.append((power[:, index] + power[:, index + 1]) / 2)
                block_starts.append(start)
                active[index:index + 2] = True
        return (cp.vstack(blocks).T if blocks else None), active, tuple(block_starts), observed_first_half

    @property
    def uses_daily_grid2_demand(self):
        return (self.rate_structure == WINTER_TOU_DEMAND_CPP
                and self.grid2_demand_formulation == "weekday_daily_dt_over_n_v1")

    @staticmethod
    def _demand_local_time(timestamp):
        timestamp = pd.Timestamp(timestamp)
        return timestamp if timestamp.tz is None else timestamp.tz_convert(LOCAL_TIMEZONE)

    def _observed_daily_demand_peaks(self, timestamp):
        """Positive net-import maxima from complete, causal 30-minute blocks.

        Meter timestamps denote interval completion. Restrict live history to
        the current local day and the controller's evaluation period; warmup
        observations do not establish a billed demand before EV activation.
        """
        timestamp = pd.Timestamp(timestamp)
        day = self._demand_local_time(timestamp).normalize()
        if day.weekday() >= 5:
            return {}
        peaks = {"on_peak_kw": 0.0, "anytime_kw": 0.0}
        for saved_day, saved in self._restored_daily_demand_peaks.items():
            if self._demand_local_time(saved_day).normalize() != day:
                continue
            for key in peaks:
                value = float(saved[key])
                if not np.isfinite(value) or value < 0:
                    raise ValueError("Restored daily demand peaks must be finite and nonnegative")
                peaks[key] = max(peaks[key], value)
        by_block = {}
        for completed, row in self._completed_plant_rows:
            completed = pd.Timestamp(completed)
            start = completed - pd.Timedelta(minutes=1)
            if completed > timestamp or self._demand_local_time(start).normalize() != day:
                continue
            if self.first_observation is not None and start < self.first_observation:
                continue
            value = float(row[TOTAL_POWER_KEY])
            if not np.isfinite(value):
                raise ValueError("Completed whole-home meter power must be finite")
            by_block.setdefault(start.floor("30min"), []).append((completed, value))
        for block_start, readings in by_block.items():
            expected = pd.date_range(block_start + pd.Timedelta(minutes=1),
                                     periods=30, freq="min")
            if expected[-1] > timestamp or [time for time, _ in readings] != list(expected):
                continue
            mean_import = max(0.0, float(np.mean([value for _, value in readings])))
            peaks["anytime_kw"] = max(peaks["anytime_kw"], mean_import)
            if WinterTOURate.is_on_peak(block_start):
                peaks["on_peak_kw"] = max(peaks["on_peak_kw"], mean_import)
        return {day.isoformat(): peaks}

    def _build_daily_grid2_demand(self, power, starts):
        """Shared daily/component scenario maxima, with fixed 0.25/N scaling."""
        blocks, block_starts = [], []
        active = np.zeros(len(starts), dtype=bool)
        observed_first_half = None
        if starts[0].minute % 30 == 15 and self._demand_local_time(starts[0]).weekday() < 5:
            block_start = starts[0] - INTERVAL
            expected = pd.date_range(block_start + pd.Timedelta(minutes=1), starts[0], freq="min")
            observed = [(time, row[TOTAL_POWER_KEY]) for time, row in self._completed_plant_rows
                        if block_start < time <= starts[0]]
            if [time for time, _ in observed] != list(expected):
                raise ValueError("Grid 2 daily demand needs 15 completed whole-home meter minutes before a mid-block decision")
            observed_first_half = float(np.mean([value for _, value in observed]))
            if not np.isfinite(observed_first_half):
                raise ValueError("Completed whole-home meter power must be finite")
            blocks.append((power[:, 0] + observed_first_half) / 2)
            block_starts.append(block_start)
            active[0] = True
        for index, start in enumerate(starts[:-1]):
            if start.minute % 30 == 0 and self._demand_local_time(start).weekday() < 5:
                blocks.append((power[:, index] + power[:, index + 1]) / 2)
                block_starts.append(start)
                active[index:index + 2] = True
        block_starts = tuple(block_starts)
        block_power = cp.vstack(blocks).T if blocks else None
        on_peak = np.array([WinterTOURate.is_on_peak(t) for t in block_starts], dtype=bool)
        local_days = tuple(dict.fromkeys(self._demand_local_time(t).normalize() for t in starts
                                        if self._demand_local_time(t).weekday() < 5))
        observed = self._observed_daily_demand_peaks(starts[0])
        observed_on = np.array([observed.get(day.isoformat(), {}).get("on_peak_kw", 0.0)
                                for day in local_days])
        observed_any = np.array([observed.get(day.isoformat(), {}).get("anytime_kw", 0.0)
                                 for day in local_days])
        scale = (INTERVAL.total_seconds() / 3600) / len(starts)
        constraints = []
        if local_days:
            on = cp.Variable(len(local_days), nonneg=True, name="ev_only_daily_on_peak_demand_kw")
            anytime = cp.Variable(len(local_days), nonneg=True, name="ev_only_daily_anytime_demand_kw")
            constraints += [on >= observed_on, anytime >= observed_any]
            for index, day in enumerate(local_days):
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
        peak_starts = tuple(t for t, selected in zip(block_starts, on_peak) if selected)
        peak_active = np.zeros(len(starts), dtype=bool)
        for lead, start in enumerate(starts):
            peak_active[lead] = start.floor("30min") in peak_starts
        return {
            "constraints": constraints,
            "daily_demand_days": local_days,
            "daily_on_peak_demand_kw": on, "daily_anytime_demand_kw": anytime,
            "daily_demand_observed_on_peak_kw": observed_on,
            "daily_demand_observed_anytime_kw": observed_any,
            "daily_demand_observed_peaks": observed,
            "daily_demand_scale": scale,
            "daily_demand_normalization_steps": len(starts),
            "daily_demand_step_hours": INTERVAL.total_seconds() / 3600,
            "daily_demand_on_peak_rate_usd_per_kw_day": GRID2_ON_PEAK_DEMAND_USD_PER_KW_DAY,
            "daily_demand_anytime_rate_usd_per_kw_day": GRID2_ANYTIME_DEMAND_USD_PER_KW_DAY,
            "daily_demand_objective_terms": costs,
            "daily_demand_block_starts": block_starts,
            "daily_demand_block_power_scenarios_kw": block_power,
            "daily_demand_block_is_on_peak": on_peak,
            "daily_demand_active": active,
            "peak_power": block_power[:, on_peak] if on_peak.any() else None,
            "peak_starts": peak_starts, "peak_active": peak_active,
            "observed_first_half": observed_first_half,
        }

    def observe_plant_step(self, timestamp: pd.Timestamp, status: Mapping[str, Any]) -> None:
        """Record only the Hub meter and EV submeter at interval completion."""
        validate_ev_only_hub_observation(
            status, self.equipment_name,
            include_total_home=self.include_total_home_meter,
        )
        completed_at = pd.Timestamp(timestamp) + pd.Timedelta(minutes=1)
        if self._completed_plant_rows and completed_at <= self._completed_plant_rows[-1][0]:
            raise ValueError("Home Hub observations must arrive once in increasing time order")
        self._completed_plant_rows.append((completed_at, dict(status)))

    def _completed_history(self, timestamp: pd.Timestamp) -> pd.DataFrame:
        rows = [(completed_at, status) for completed_at, status in self._completed_plant_rows
                if completed_at <= timestamp]
        history = pd.DataFrame(
            [status for _, status in rows],
            index=pd.DatetimeIndex([time for time, _ in rows], name="completed_at"),
            columns=observation_keys(
                self.equipment_name,
                include_total_home=self.include_total_home_meter,
            ),
        )
        if not history.empty and history.index[-1] > timestamp:
            raise AssertionError("Future Home Hub observation entered a forecast issuance")
        return history

    def _issue_hub(self, timestamp: pd.Timestamp) -> EVHubIntervalForecast:
        history = self._completed_history(timestamp)
        issued = self.hub_forecast_provider.issue(
            timestamp, history, self.horizon, self.options.max_discharge_kw,
        )
        if not isinstance(issued, EVHubIntervalForecast):
            raise TypeError("Home Hub provider must return EVHubIntervalForecast")
        issued = replace(issued, mobility_split_policy=self.hub_mobility_split_policy)
        issued.window(timestamp, self.horizon, self.options.max_discharge_kw)
        self._issued_hub = issued
        self.issued_load_history.append({
            "timestamp": timestamp,
            "hub_source": issued.source,
            "hub_inputs_ready_at": issued.inputs_ready_at,
            "hub_mobility_split_policy": issued.mobility_split_policy,
            "latest_completed_plant_observation": history.index[-1] if not history.empty else None,
        })
        return issued

    def build_problem(self, timestamp, soc, forecast):
        """Retain EV scenario costs and add one worst-scenario home peak term."""
        if not isinstance(forecast, EVScenarioSet) or len(forecast.forecasts) != self.scenario_count:
            raise ValueError(f"EV Home Hub control requires {self.scenario_count} EV mobility scenarios")
        timestamp = pd.Timestamp(timestamp)
        problem, plan = super().build_problem(timestamp, soc, forecast)
        starts = pd.date_range(timestamp, periods=int(self.horizon / INTERVAL), freq=INTERVAL)
        issued = forecast.forecasts[0].ev_hub_intervals
        if issued is None or any(path.ev_hub_intervals is not issued for path in forecast.forecasts):
            raise ValueError("EV paths must share one issued Home Hub forecast")
        hub = issued.window(timestamp, self.horizon, self.options.max_discharge_kw)
        if getattr(self.outside_hub_forecast, "requires_total_home_meter", False):
            outside = self.outside_hub_forecast.forecast_from_completed(
                timestamp, self._completed_history(timestamp), steps=len(starts),
            )
        else:
            outside = self.outside_hub_forecast.forecast_kw(timestamp, steps=len(starts))
        if (not isinstance(outside, pd.Series) or not outside.index.equals(starts)
                or not np.isfinite(outside.to_numpy(dtype=float)).all() or (outside < 0).any()):
            raise ValueError("Outside-Hub forecast must provide aligned nonnegative 15-minute values covering the horizon")
        ev_power = cp.vstack([
            _mean_ev_power_on_intervals(path, starts) for path in plan["scenario_plans"]
        ])
        non_ev_home_kw = hub["hub_mean_kw"].to_numpy(dtype=float) + outside.to_numpy(dtype=float)
        modeled_home = ev_power + np.broadcast_to(non_ev_home_kw, (len(plan["scenario_plans"]), len(starts)))
        block_starts, observed_first_half, daily_demand = (), None, None
        if self.uses_daily_grid2_demand:
            daily_demand = self._build_daily_grid2_demand(modeled_home, starts)
            peak_power = daily_demand["peak_power"]
            active = daily_demand["peak_active"]
            block_starts = daily_demand["peak_starts"]
            observed_first_half = daily_demand["observed_first_half"]
        elif self.peak_window == "winter_tou":
            peak_power, active, block_starts, observed_first_half = self._winter_demand_blocks(modeled_home, starts)
        elif self.peak_window == "all":
            active = np.ones(len(starts), dtype=bool)
        elif self.peak_window == "evening":
            active = np.asarray((starts.hour >= 16) & (starts.hour < 22), dtype=bool)
        else:
            active = np.asarray((starts.hour >= 3) & (starts.hour < 7), dtype=bool)
        if self.peak_window != "winter_tou":
            active &= self.peak_weight_usd_per_kw > 0
            peak_power = modeled_home[:, active] if active.any() else None
        constraints = list(problem.constraints)
        if daily_demand is not None:
            constraints.extend(daily_demand["constraints"])
            peak_kw = None
            peak_cost = sum(daily_demand["daily_demand_objective_terms"].values())
        elif peak_power is not None:
            peak_kw = cp.Variable(
                nonneg=True,
                name=("ev_only_modeled_home_peak_exceedance_kw" if self.peak_limit_kw is not None
                      else "ev_only_modeled_home_peak_kw"),
            )
            if self.peak_limit_kw is None:
                constraints.append(peak_kw >= peak_power)
                peak_cost = self.peak_weight_usd_per_kw * peak_kw
            else:
                constraints.append(peak_kw >= peak_power - self.peak_limit_kw)
                # Match the 15-minute electricity objective's Δt/N scaling.
                peak_cost = (self.peak_weight_usd_per_kw
                             * (INTERVAL.total_seconds() / 3600) / len(starts) * peak_kw)
        else:
            peak_kw = None
            peak_cost = cp.Constant(0.0)
        terms = {**plan["objective_terms"], "modeled_home_peak": peak_cost}
        panel, panel_slack, grid_slack = None, None, None
        grid_active = np.zeros(len(starts), dtype=bool)
        if self.hard_panel_limit_kw is not None:
            ev_charge = cp.vstack([
                _mean_ev_power_on_intervals(path, starts, power_key="charge")
                for path in plan["scenario_plans"]
            ])
            panel = ev_charge + np.broadcast_to(non_ev_home_kw, modeled_home.shape)
            constraints.append(panel <= self.hard_panel_limit_kw)
        elif self.rate_structure == MISO_RTP_DOE:
            ev_charge = cp.vstack([
                _mean_ev_power_on_intervals(path, starts, power_key="charge")
                for path in plan["scenario_plans"]
            ])
            panel = ev_charge + np.broadcast_to(non_ev_home_kw, modeled_home.shape)
            panel_slack = cp.Variable(len(starts), nonneg=True, name="ev_only_doe_panel_slack_kw")
            constraints.append(panel <= 38.0 + panel_slack)
            terms["doe_panel_violation"] = 1000.0 * 0.25 / len(starts) * cp.sum(panel_slack)
            local = starts if starts.tz is None else starts.tz_convert(LOCAL_TIMEZONE)
            grid_active = np.asarray((local.dayofweek < 5) & (
                ((local.hour >= 6) & (local.hour < 9)) | ((local.hour >= 17) & (local.hour < 21))
            ), dtype=bool)
            if grid_active.any():
                grid_slack = cp.Variable(int(grid_active.sum()), nonneg=True, name="ev_only_doe_grid_slack_kw")
                constraints.append(modeled_home[:, grid_active] <= 15.0 + grid_slack)
                terms["doe_grid_violation"] = 10.0 * 0.25 / len(starts) * cp.sum(grid_slack)
            else:
                terms["doe_grid_violation"] = cp.Constant(0.0)
        combined = cp.Problem(cp.Minimize(sum(terms.values())), constraints)
        if not combined.is_dcp() or not combined.is_mixed_integer():
            raise ValueError("EV Home Hub scenario formulation must be a DCP-compliant MILP")
        plan.update(objective_terms=terms,
                    modeled_home_peak_kw=peak_kw if self.peak_limit_kw is None else None,
                    modeled_home_peak_exceedance_kw=peak_kw if self.peak_limit_kw is not None else None,
                    peak_limit_kw=self.peak_limit_kw, peak_window=self.peak_window,
                    modeled_home_power_scenarios_kw=modeled_home,
                    modeled_ev_power_scenarios_kw=ev_power,
                    outside_hub_forecast_kw=outside,
                    hub_forecast_kw=hub["hub_mean_kw"], peak_objective_active=active,
                    rate_structure=self.rate_structure,
                    peak_demand_block_power_scenarios_kw=peak_power if block_starts else None,
                    peak_demand_block_starts=block_starts,
                    peak_observed_first_half_kw=observed_first_half,
                    modeled_panel_power_scenarios_kw=panel,
                    hard_panel_limit_kw=self.hard_panel_limit_kw,
                    doe_panel_slack_kw=panel_slack, doe_grid_slack_kw=grid_slack,
                    doe_grid_active=grid_active)
        plan["grid2_demand_formulation"] = self.grid2_demand_formulation
        if daily_demand is not None:
            plan.update({key: value for key, value in daily_demand.items()
                         if key.startswith("daily_")})
        # Preserve the complete issue-time forecast on the same grid used by
        # the peak epigraph. Logging adds no variables, constraints, or costs.
        issued_rows = hub.loc[starts, list(hub.columns)].copy()
        issued_rows.insert(0, "target_interval_start", starts)
        issued_rows.insert(0, "lead_step", np.arange(len(starts), dtype=int))
        issued_rows.insert(0, "issued_at", timestamp)
        issued_rows["hub_source"] = issued.source
        issued_rows["hub_inputs_ready_at"] = issued.inputs_ready_at
        issued_rows["hub_mobility_split_policy"] = issued.mobility_split_policy
        issued_rows["outside_method"] = getattr(self.outside_hub_forecast, "method_label", None)
        issued_rows["outside_prior_issue_fallback"] = bool(
            outside.attrs.get("prior_issue_fallback", False)
        )
        issued_rows["outside_actual_issue_time"] = outside.attrs.get("actual_issue_time", timestamp)
        issued_rows["outside_hub_kw"] = outside.to_numpy(dtype=float)
        for column in (
            "persisted_ambient_temperature_c",
            "persisted_ambient_temperature_available",
            "persisted_ambient_relative_humidity",
            "persisted_ambient_relative_humidity_available",
            "weather_inputs_ready_at",
            "persisted_weather_window_start",
            "persisted_weather_window_end_exclusive",
        ):
            issued_rows[column] = outside.attrs.get(column, np.nan)
        issued_rows["non_ev_home_kw"] = non_ev_home_kw
        issued_rows["peak_penalty_active"] = active
        issued_rows["peak_limit_kw"] = self.peak_limit_kw
        issued_rows["rate_structure"] = self.rate_structure
        if daily_demand is not None:
            issued_rows["grid2_demand_formulation"] = self.grid2_demand_formulation
            issued_rows["daily_anytime_demand_active"] = daily_demand["daily_demand_active"]
            issued_rows["daily_demand_scale"] = daily_demand["daily_demand_scale"]
        issued_rows["doe_grid_active"] = grid_active
        issued_rows["doe_grid_limit_kw"] = 15.0 if self.rate_structure == MISO_RTP_DOE else np.nan
        issued_rows["doe_panel_limit_kw"] = 38.0 if self.rate_structure == MISO_RTP_DOE else np.nan
        price = forecast.forecasts[0].data.reindex(starts).get(PRICE_COLUMN)
        issued_rows["tou_price_usd_per_kwh"] = (
            np.nan if price is None else price.to_numpy(dtype=float)
        )
        self.forecast_trace.append(issued_rows.reset_index(drop=True))
        if self.trace_callback is not None:
            self.trace_callback("forecast_inputs", self.forecast_trace[-1])
        # Mobility scenario forecasts can split a 15-minute block at a trip
        # boundary. Save their native segments before the solve, including the
        # exact load-following and tariff inputs used by the EV MILP.
        paths = plan["scenario_plans"]
        if all("horizon" in path for path in paths):
            identifiers = plan.get("scenario_ids") or tuple(str(i) for i in range(len(paths)))
            scenario_inputs = []
            snapshots = []
            for index, (identifier, path) in enumerate(zip(identifiers, paths)):
                h = path["horizon"]
                mobility = forecast.forecasts[index].ev_schedules[self.ev.vehicle_id]
                snapshots.append({
                    "issued_at": timestamp, "scenario_index": index, "scenario_id": identifier,
                    "scenario_probability": 1 / len(paths),
                    "mobility_start": mobility.start, "mobility_end": mobility.end,
                    # pandas' table JSON schema assumes pytz's `.zone`, which
                    # ZoneInfo does not expose. Record strings retain local
                    # offsets and every trip column, including zero demand.
                    "mobility_trips_json": json.dumps(mobility.trips.to_dict("records"), default=str),
                    "mobility_trip_columns_json": json.dumps(mobility.trips.columns.tolist()),
                    "mobility_metadata_json": json.dumps(mobility.metadata, default=str),
                    "terminal_value_usd_per_kwh": path["terminal_value_usd_per_kwh"],
                    "horizon_json": json.dumps({
                        key: (value.astype(str).tolist() if isinstance(value, pd.DatetimeIndex)
                              else value.tolist() if isinstance(value, np.ndarray) else value)
                        for key, value in vars(h).items()
                    }, default=str),
                })
                for segment in range(len(h.dt)):
                    scenario_inputs.append({
                        "issued_at": timestamp,
                        "scenario_index": index,
                        "scenario_id": identifier,
                        "scenario_probability": 1 / len(paths),
                        "segment_index": segment,
                        "segment_start": h.t[segment],
                        "segment_end": h.t[segment + 1],
                        "duration_hours": float(h.dt[segment]),
                        "ev_available": bool(h.z[segment]),
                        "trip_energy_kwh": float(h.trip_kwh[segment]),
                        "trip_shortfall_eligible": bool(h.trip_kwh[segment] > 0),
                        "tou_price_usd_per_kwh": float(h.price[segment]),
                        "eligible_discharge_mean_kw": float(h.discharge_mean[segment]),
                        "eligible_discharge_min_kw": float(h.discharge_min[segment]),
                        "fixed_backed_tariff_usd_per_hour": float(h.backed_cost[segment]),
                    })
            self.ev_scenario_input_trace.append(pd.DataFrame(scenario_inputs))
            if self.trace_callback is not None:
                self.trace_callback("ev_scenario_inputs", self.ev_scenario_input_trace[-1])
                self.trace_callback("scenario_snapshot", pd.DataFrame(snapshots))
        self._last_built_plan = plan
        return combined, plan

    def _solve_problem(self, problem, options, *, time_limit_seconds, allow_unlimited_time):
        """Compile the unchanged EV graph, retaining ordinary EV acceptance."""
        self._solver_problem = problem
        self._solver_history_length = len(self.solve_history)
        callback = self._solve_sparse_problem if self.solver_backend == "sparse_native_v1" else None
        try:
            solve_ocp(problem, self.options.solver, options,
                      time_limit_seconds=time_limit_seconds,
                      allow_unlimited_time=allow_unlimited_time,
                      solve_callback=callback)
        finally:
            diagnostic = getattr(problem, "hems_diagnostic", {})
            diagnostic.setdefault("solver_backend", "cvxpy")
            diagnostic["requested_solver_backend"] = self.solver_backend
            problem.hems_diagnostic = diagnostic

    def _solve_sparse_problem(self, problem, *, solver, **options):
        """Use one native model; unsupported graphs use the original compiler."""
        started = time.perf_counter()
        diagnostic = problem.hems_diagnostic
        native_options = {key: value for key, value in options.items()
                          if key not in {"verbose", "solver_verbose"}}
        if self._fast_solver is None:
            self._fast_solver = ReusableJointMILPSolver()
        self._fast_solver.last_metrics = {}
        try:
            if {"env", "reoptimize", "warm_start"}.intersection(native_options):
                raise UnsupportedFastMILP("CVXPY-specific solver options require the CVXPY backend")
            diagnostic["solver_backend"] = "sparse_native_v1"
            self._fast_solver.solve(problem, native_options)
            diagnostic["fast_solver_metrics"] = dict(self._fast_solver.last_metrics)
        except UnsupportedFastMILP as error:
            self._fast_solver.close()
            diagnostic.update(solver_backend="cvxpy", solver_backend_fallback_reason=str(error))
            if "TimeLimit" in options:
                options["TimeLimit"] -= time.perf_counter() - started
                if options["TimeLimit"] <= 0:
                    diagnostic.update(timed_out=True, limit_reached=True,
                                      outer_status="time_budget_exhausted_before_fallback")
                    raise cp.error.SolverError("EV solver time budget exhausted before compiler fallback")
                diagnostic["backend_limit_seconds"] = options["TimeLimit"]
            problem.solve(solver=solver, **options)
        finally:
            if diagnostic.get("solver_backend") == "sparse_native_v1":
                diagnostic["fast_solver_metrics"] = dict(self._fast_solver.last_metrics)

    def _release_solver_problem(self):
        """Snapshot native quality before disposing or reusing its model."""
        problem, self._solver_problem = self._solver_problem, None
        if problem is None:
            return
        stats = getattr(problem, "solver_stats", None)
        model = getattr(stats, "extra_stats", None)
        if model is None and self._fast_solver is not None:
            optimized = getattr(self._fast_solver, "last_optimized_problem_ref", None)
            if optimized is not None and optimized() is problem:
                # Matrix validation or unpack can reject a fresh optimized
                # candidate before CVXPY receives SolverStats.
                model = self._fast_solver.model
        quality = {}
        if callable(getattr(model, "dispose", None)):
            for name in ("Status", "SolCount", "ObjVal", "ObjBound", "MIPGap", "Runtime",
                         "MaxVio", "BoundVio", "ConstrVio", "ConstrResidual", "IntVio"):
                try:
                    quality[name] = getattr(model, name)
                except Exception:
                    pass
            retained = self._fast_solver is not None and model is self._fast_solver.model
            if not retained:
                model.dispose()
        if len(self.solve_history) > self._solver_history_length:
            self.solve_history[-1]["native_solver_quality"] = quality
        problem._solver_cache.clear()

    def close(self):
        """Release the sole retained Gurobi model after a replay or benchmark."""
        self._release_solver_problem()
        if self._fast_solver is not None:
            self._fast_solver.close()

    def _solve(self, timestamp, soc, forecast):
        try:
            self._solve_and_record(timestamp, soc, forecast)
        finally:
            self._release_solver_problem()

    def _solve_and_record(self, timestamp, soc, forecast):
        if self.one_shot and self._one_shot_attempted:
            raise RuntimeError("The full-period EV optimization may only be solved once")
        self._one_shot_attempted = self.one_shot
        if self.trace_callback is not None:
            self.trace_callback("decision_state", pd.DataFrame([{
                "issued_at": pd.Timestamp(timestamp), "soc": soc,
                "soh": self.current_soh, "capacity_kwh": self.current_capacity,
                "previous_mode": self.previous_mode, "previous_power_kw": self.previous_power,
                "previous_chemical_power_kw": self.previous_chemical_power,
                "held_mode": self.held_mode, "previous_target_kw": self.target,
                "first_observation": self.first_observation,
                "soc_turning_points_json": json.dumps(self.soc_history),
                "discharge_samples_json": json.dumps(self.discharge_samples, default=str),
                "mode_starts_json": json.dumps(self.starts, default=str),
                "mode_ends_json": json.dumps(self.ends, default=str),
                "last_change_json": json.dumps(self.last_change, default=str),
                "completed_plant_rows_json": json.dumps(self._completed_plant_rows[-15:], default=str),
                "rate_structure": self.rate_structure, "one_shot": self.one_shot,
                "grid2_demand_formulation": self.grid2_demand_formulation,
                "observed_daily_demand_peaks_json": json.dumps(
                    self._observed_daily_demand_peaks(timestamp) if self.uses_daily_grid2_demand else {}),
                "normalization_steps": int(self.horizon / INTERVAL),
            }]))
        self._last_built_plan = None
        super()._solve(timestamp, soc, forecast)
        self._release_solver_problem()
        plan = self._last_built_plan
        latest = self.solve_history[-1]
        latest.setdefault("solver_backend", "cvxpy")
        latest.setdefault("requested_solver_backend", self.solver_backend)
        if (self.solver_backend == "sparse_native_v1" and plan is not None
                and plan["rainflow_theta"] is not None):
            latest.setdefault("solver_backend_fallback_reason", "Preserve certified CVXPY rainflow master loop")
        if latest["successful"] and plan is not None:
            # The existing electricity term includes a fixed backed-load
            # tariff constant. Split it for interpretation only; the solved
            # expression above is unchanged. Scenario costs enter by mean.
            if self.plans and pd.Timestamp(self.plans[-1]["timestamp"]) == pd.Timestamp(timestamp):
                scenario_splits = []
                for path in plan["scenario_plans"]:
                    h = path["horizon"]
                    fixed = self.options.electricity_weight * float(
                        np.dot(h.dt, h.backed_cost) / h.N
                    )
                    charge = self.options.electricity_weight * float(
                        np.dot(h.dt * h.price, np.asarray(path["charge"].value, dtype=float)) / h.N
                    )
                    total = float(path["objective_terms"]["electricity"].value)
                    scenario_splits.append({
                        "fixed_backed_load_cost": fixed,
                        "ev_charge_cost": charge,
                        "ev_discharge_credit": total - fixed - charge,
                    })
                for saved, split in zip(self.plans[-1]["scenarios"], scenario_splits):
                    saved["electricity_components"] = split
                self.plans[-1]["electricity_components"] = {
                    key: float(np.mean([entry[key] for entry in scenario_splits]))
                    for key in scenario_splits[0]
                }
                peak_var = (plan["modeled_home_peak_exceedance_kw"] if self.peak_limit_kw is not None
                            else plan["modeled_home_peak_kw"])
                self.plans[-1]["modeled_home_peak_epigraph_kw"] = (
                    float(peak_var.value) if peak_var is not None else 0.0
                )
                self.plans[-1].update(
                    rate_structure=self.rate_structure, one_shot=self.one_shot,
                    grid2_demand_formulation=self.grid2_demand_formulation,
                    peak_demand_block_starts=[str(value) for value in plan["peak_demand_block_starts"]],
                    peak_observed_first_half_kw=plan["peak_observed_first_half_kw"],
                    peak_demand_block_power_scenarios_kw=(
                        None if plan["peak_demand_block_power_scenarios_kw"] is None
                        else plan["peak_demand_block_power_scenarios_kw"].value.tolist()
                    ),
                    doe_panel_slack_kw=(None if plan["doe_panel_slack_kw"] is None
                                        else plan["doe_panel_slack_kw"].value.tolist()),
                    doe_grid_slack_kw=(None if plan["doe_grid_slack_kw"] is None
                                       else plan["doe_grid_slack_kw"].value.tolist()),
                    doe_grid_active=plan["doe_grid_active"].tolist(),
                )
                if self.uses_daily_grid2_demand:
                    daily_fields = {
                        "daily_demand_days": [day.isoformat() for day in plan["daily_demand_days"]],
                        "daily_demand_objective_terms": {
                            key: float(value.value) for key, value in plan["daily_demand_objective_terms"].items()
                        },
                        "daily_demand_observed_peaks": plan["daily_demand_observed_peaks"],
                        "daily_demand_block_starts": [str(value) for value in plan["daily_demand_block_starts"]],
                        "daily_demand_block_power_scenarios_kw": (
                            None if plan["daily_demand_block_power_scenarios_kw"] is None
                            else plan["daily_demand_block_power_scenarios_kw"].value.tolist()
                        ),
                    }
                    for key in ("daily_on_peak_demand_kw", "daily_anytime_demand_kw"):
                        daily_fields[key] = None if plan[key] is None else plan[key].value.tolist()
                    for key in ("daily_demand_observed_on_peak_kw", "daily_demand_observed_anytime_kw",
                                "daily_demand_block_is_on_peak", "daily_demand_active"):
                        daily_fields[key] = plan[key].tolist()
                    for key in ("daily_demand_scale", "daily_demand_step_hours",
                                "daily_demand_normalization_steps", "daily_demand_on_peak_rate_usd_per_kw_day",
                                "daily_demand_anytime_rate_usd_per_kw_day"):
                        daily_fields[key] = plan[key]
                    self.plans[-1].update(daily_fields)
                for saved, path in zip(self.plans[-1]["scenarios"], plan["scenario_plans"]):
                    saved["charge_mode"] = np.asarray(path["charge_mode"].value > .5, dtype=int).tolist()
                    saved["discharge_mode"] = np.asarray(path["discharge_mode"].value > .5, dtype=int).tolist()
                    saved.update(predicted_trip_energy_fields(path))
                if self.one_shot:
                    path = plan["scenario_plans"][0]
                    self._one_shot_boundaries = pd.DatetimeIndex(path["boundaries"])
                    self._one_shot_modes = (
                        np.asarray(path["charge_mode"].value > .5, dtype=int)
                        - np.asarray(path["discharge_mode"].value > .5, dtype=int)
                    )
                    self.plans[-1]["one_shot_modes"] = self._one_shot_modes.tolist()
                    # Persist the sole accepted schedule immediately, including
                    # states and binaries, before any minute of replay.
                    if self.trace_callback is not None:
                        self.trace_callback("accepted_plan", pd.DataFrame([{
                            "issued_at": pd.Timestamp(timestamp),
                            "plan_json": json.dumps(self.plans[-1], default=str),
                        }]))
            home = np.asarray(plan["modeled_home_power_scenarios_kw"].value, dtype=float)
            ev = np.asarray(plan["modeled_ev_power_scenarios_kw"].value, dtype=float)
            starts = pd.date_range(pd.Timestamp(timestamp), periods=home.shape[1], freq=INTERVAL)
            panel = (None if plan["modeled_panel_power_scenarios_kw"] is None
                     else np.asarray(plan["modeled_panel_power_scenarios_kw"].value, dtype=float))
            panel_slack = (np.zeros(len(starts)) if plan["doe_panel_slack_kw"] is None
                           else np.asarray(plan["doe_panel_slack_kw"].value, dtype=float))
            grid_slack = np.zeros(len(starts))
            if plan["doe_grid_slack_kw"] is not None:
                grid_slack[plan["doe_grid_active"]] = plan["doe_grid_slack_kw"].value
            demand_power = (None if plan["peak_demand_block_power_scenarios_kw"] is None
                            else np.asarray(plan["peak_demand_block_power_scenarios_kw"].value, dtype=float))
            demand_block_index = {start: index for index, start in enumerate(plan["peak_demand_block_starts"])}
            peak_epigraph = (plan["modeled_home_peak_exceedance_kw"] if self.peak_limit_kw is not None
                             else plan["modeled_home_peak_kw"])
            peak_value = 0.0 if peak_epigraph is None else float(peak_epigraph.value)
            daily_indices = ({day: index for index, day in enumerate(plan["daily_demand_days"])}
                             if self.uses_daily_grid2_demand else {})
            daily_blocks = ({start: index for index, start in enumerate(plan["daily_demand_block_starts"])}
                            if self.uses_daily_grid2_demand else {})
            scenario_ids = plan.get("scenario_ids") or tuple(str(i) for i in range(home.shape[0]))
            rows = []
            for index, identifier in enumerate(scenario_ids):
                path = plan["scenario_plans"][index]
                h = path.get("horizon")
                trip_fields = predicted_trip_energy_fields(path) if h is not None else None
                for lead, target in enumerate(starts):
                    block_start = target.floor("30min")
                    demand_index = demand_block_index.get(block_start)
                    demand_kw = (np.nan if demand_index is None else float(demand_power[index, demand_index]))
                    peak_lhs = demand_kw if self.peak_window == "winter_tou" else float(home[index, lead])
                    daily_index = daily_indices.get(self._demand_local_time(target).normalize())
                    row_peak_value = (float(plan["daily_on_peak_demand_kw"].value[daily_index])
                                      if daily_index is not None else peak_value)
                    prediction_row = {
                        "issued_at": pd.Timestamp(timestamp),
                        "scenario_index": index,
                        "scenario_id": identifier,
                        "scenario_probability": 1 / len(scenario_ids),
                        "lead_step": lead,
                        "target_interval_start": target,
                        "ev_power_kw": float(ev[index, lead]),
                        "modeled_home_power_kw": float(home[index, lead]),
                        "peak_penalty_active": bool(plan["peak_objective_active"][lead]),
                        "peak_limit_kw": self.peak_limit_kw,
                        "rate_structure": self.rate_structure,
                        "peak_epigraph_kw": row_peak_value,
                        "peak_constraint_margin_kw": (
                            row_peak_value + (self.peak_limit_kw or 0.0) - peak_lhs
                            if plan["peak_objective_active"][lead] else np.nan
                        ),
                        "demand_block_start": block_start if demand_index is not None else None,
                        "demand_block_power_kw": demand_kw,
                        "peak_observed_first_half_kw": plan["peak_observed_first_half_kw"],
                        "doe_grid_active": bool(plan["doe_grid_active"][lead]),
                        "doe_grid_limit_kw": 15.0 if self.rate_structure == MISO_RTP_DOE else np.nan,
                        "doe_grid_slack_kw": float(grid_slack[lead]),
                        "doe_grid_constraint_margin_kw": (
                            15.0 + grid_slack[lead] - float(home[index, lead])
                            if plan["doe_grid_active"][lead] else np.nan
                        ),
                        "doe_panel_gross_kw": np.nan if panel is None else float(panel[index, lead]),
                        "doe_panel_limit_kw": 38.0 if panel is not None else np.nan,
                        "doe_panel_slack_kw": float(panel_slack[lead]),
                        "doe_panel_constraint_margin_kw": (
                            np.nan if panel is None else 38.0 + panel_slack[lead] - float(panel[index, lead])
                        ),
                        "active_positive_exceedance_kw": (
                            max(0.0, float(home[index, lead]) - self.peak_limit_kw)
                            if self.peak_limit_kw is not None and plan["peak_objective_active"][lead]
                            else 0.0
                        ),
                    }
                    if self.uses_daily_grid2_demand:
                        daily_block_index = daily_blocks.get(block_start)
                        daily_power = (np.nan if daily_block_index is None else float(
                            plan["daily_demand_block_power_scenarios_kw"].value[index, daily_block_index]))
                        anytime_value = (np.nan if daily_index is None else float(
                            plan["daily_anytime_demand_kw"].value[daily_index]))
                        prediction_row.update(
                            grid2_demand_formulation=self.grid2_demand_formulation,
                            daily_demand_day=(None if daily_index is None
                                              else plan["daily_demand_days"][daily_index].isoformat()),
                            daily_demand_scale=plan["daily_demand_scale"],
                            daily_on_peak_demand_kw=(np.nan if daily_index is None else row_peak_value),
                            daily_anytime_demand_kw=anytime_value,
                            daily_demand_block_power_kw=daily_power,
                            daily_anytime_demand_active=bool(plan["daily_demand_active"][lead]),
                            daily_anytime_constraint_margin_kw=anytime_value - daily_power,
                            daily_on_peak_observed_floor_kw=(np.nan if daily_index is None else float(
                                plan["daily_demand_observed_on_peak_kw"][daily_index])),
                            daily_anytime_observed_floor_kw=(np.nan if daily_index is None else float(
                                plan["daily_demand_observed_anytime_kw"][daily_index])),
                        )
                    if trip_fields is not None:
                        segments = np.flatnonzero(h.group == lead)
                        prediction_row["trip_shortfall_eligible"] = bool(np.any(h.trip_kwh[segments] > 0))
                        for key in ("requested_trip_energy_kwh", "predicted_trip_shortfall_kwh",
                                    "predicted_served_trip_energy_kwh"):
                            prediction_row[key] = float(np.asarray(trip_fields[key])[segments].sum())
                        # Keep exact event allocation when a mobility boundary
                        # splits a control interval into several native blocks.
                        prediction_row["trip_energy_segments_json"] = json.dumps([
                            {"segment_index": int(segment),
                             "segment_start": str(h.t[segment]),
                             "segment_end": str(h.t[segment + 1]),
                             "trip_shortfall_eligible": bool(h.trip_kwh[segment] > 0),
                             **{key: trip_fields[key][segment] for key in (
                                 "requested_trip_energy_kwh", "predicted_trip_shortfall_kwh",
                                 "predicted_served_trip_energy_kwh",
                             )}}
                            for segment in segments
                        ])
                    rows.append(prediction_row)
            self.scenario_prediction_trace.append(pd.DataFrame(rows))
            if self.trace_callback is not None:
                self.trace_callback("scenario_predictions", self.scenario_prediction_trace[-1])
        if self.trace_callback is not None:
            from ..ev_home_hub_trace import objective_component_frames

            saved = (self.plans[-1:] if latest["successful"] else [])
            objective, scenario_objective = objective_component_frames(saved, [latest])
            self.trace_callback("objective_components", objective)
            if not scenario_objective.empty:
                self.trace_callback("scenario_objective_components", scenario_objective)
        if self.progress_callback is not None:
            self.progress_callback({
                "timestamp": str(timestamp),
                "solve_count": len(self.solve_history),
                "successful_solves": sum(bool(row["successful"]) for row in self.solve_history),
                "fallback_count": sum(bool(row["fallback"]) for row in self.solve_history),
                "last_status": latest["status"],
                "last_solve_seconds": latest["solve_seconds"],
                "last_fallback": latest["fallback"],
                "latest_solve_row": dict(latest),
            })
        if self.one_shot and not latest["successful"]:
            raise RuntimeError(f"One-shot EV optimization did not produce an accepted schedule: {latest['status']}")

    def controls_for_step(
        self, timestamp: pd.Timestamp, previous_status: Mapping[str, Any] | None,
        current_schedule: Mapping[str, Any], forecast: Forecast,
    ) -> dict[str, dict[str, Any]]:
        """Hold charge/idle/discharge mode; the plant clips full V2H each minute."""
        timestamp = pd.Timestamp(timestamp)
        if forecast.backed_load_kw is not None:
            raise ValueError("EV Home Hub control cannot use a minute backed-load forecast")
        if previous_status is not None:
            validate_ev_only_hub_observation(
                previous_status, self.equipment_name,
                include_total_home=self.include_total_home_meter,
            )
        if set(current_schedule) - {PRICE_COLUMN}:
            raise ValueError("EV Home Hub control cannot receive equipment schedule inputs")
        if set(forecast.data.columns) != {PRICE_COLUMN}:
            raise ValueError("EV Home Hub forecast data must contain only the tariff")
        if (forecast.water_scenarios is not None or forecast.rest_of_home_forecast is not None
                or forecast.weather_archive is not None):
            raise ValueError("EV Home Hub control cannot receive thermal or other-load context")
        if set(forecast.providers) - {self.equipment_name}:
            raise ValueError("EV Home Hub control cannot receive other device providers")
        if set(forecast.ev_schedules) - {self.ev.vehicle_id}:
            raise ValueError("EV Home Hub control cannot receive other vehicles' schedules")
        if set(forecast.ev_states) - set(observation_keys(self.equipment_name)):
            raise ValueError("EV Home Hub control cannot receive other equipment states")
        status = dict(previous_status or {})
        status.update(forecast.ev_states)
        soc = float(np.clip(status[f"{self.equipment_name} SOC (-)"], 0, 1))
        mobility = forecast.ev_schedules[self.ev.vehicle_id]
        available = bool(mobility.available([timestamp])[0])
        step = forecast.data.index[1] - forecast.data.index[0]
        self._observe(timestamp, float(status.get(f"{self.equipment_name} Electric Power (kW)", 0)), step)
        soh = float(status[f"{self.equipment_name} SOH (-)"])
        if not np.isfinite(soh) or not 0 < soh <= 1:
            raise ValueError("EV measured SOH must be in (0,1]")
        self.current_soh = soh
        self.current_capacity = self.ev.capacity_kwh * soh
        reported_capacity = status.get(f"{self.equipment_name} Capacity (kWh)", self.current_capacity)
        if not np.isclose(reported_capacity, self.current_capacity, rtol=1e-9, atol=1e-10):
            raise ValueError("EV measured SOH and capacity are inconsistent")
        self.seed_observed_peak_comfort(
            timestamp, mobility, forecast, self._completed_plant_rows,
        )
        self.observe_realized_peak_comfort(
            timestamp, mobility, soc, forecast,
            parked=status.get(f"{self.equipment_name} Parked"),
        )
        self.soc_history.extend([float(status.get(f"{self.equipment_name} End SOC (-)", soc)), soc])
        self.soc_history = np.asarray(self.soc_history)[turning_points(self.soc_history)].tolist()
        if ((self.one_shot and not self._one_shot_attempted)
                or (not self.one_shot and (self.next_solve is None or timestamp >= self.next_solve))):
            issued = self._issue_hub(timestamp)
            self._solve(timestamp, soc, replace(forecast, ev_hub_intervals=issued))
        if self.one_shot:
            if self._one_shot_boundaries is None or self._one_shot_modes is None:
                raise RuntimeError("One-shot EV control has no accepted schedule to replay")
            index = int(self._one_shot_boundaries.searchsorted(timestamp, side="right")) - 1
            if index < 0 or index >= len(self._one_shot_modes):
                raise ValueError("One-shot EV schedule does not cover the requested plant minute")
            self.held_mode = int(self._one_shot_modes[index])
        if self.fallback:
            # Preserve the existing EV fallback when no acceptable solve exists.
            target = (min(self.ev.charger_power_kw,
                          max(0, self.comfort_target_at(timestamp) - soc) * self.current_capacity /
                          (step.total_seconds() / 3600 * self.options.charge_efficiency))
                      if available else 0.0)
            mode, power, override = int(target > 0), target, False
        else:
            target = (self.ev.charger_power_kw if self.held_mode == 1 else
                      -self.options.max_discharge_kw if self.held_mode == -1 else 0.0)
            mode, power, override = self._project(
                timestamp, target, soc, available, self.options.max_discharge_kw,
            )
        self.action_history.append(dict(timestamp=timestamp, requested_kw=target, projected_kw=power,
                                        mode=mode, planned_mode=self.held_mode, one_shot=self.one_shot,
                                        projection_override=override, fallback=self.fallback))
        # Zero is intentional: without the OCHRE actual_backed_load callback a
        # missed plant integration cannot allow unsupported V2H discharge.
        return {self.equipment_name: {"P Setpoint": power, "Backed Load (kW)": 0.0}}
