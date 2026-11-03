"""Simple state-space predictive controller for the water heater."""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import cvxpy as cp
import numpy as np
import pandas as pd

from ..forecast import PRICE_COLUMN, Forecast
from ..water_heater_rc import WaterHeaterRCModel, rc_step_matrices
from .water_heater_scenario import build_scenario_problem


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[2] / "models" / "water_heater_model"
)
DEFAULT_RC_MODEL_PATH = (
    Path(__file__).resolve().parents[2] / "models" / "water_heater_rc_model"
)
DEFAULT_RC_OLD_MODEL_PATH = DEFAULT_RC_MODEL_PATH.parent / "water_heater_rc_model_old"
DEFAULT_RC_STRATIFIED_MODEL_PATH = DEFAULT_RC_MODEL_PATH.parent / "water_heater_rc_model_stratified"


@dataclass(frozen=True)
class PreparedWaterHeaterScenarioStep:
    """One built scenario problem, ready for a standalone or joint solve."""

    timestamp: pd.Timestamp
    problem: cp.Problem
    heat: cp.Expression
    electric_power_kw: cp.Expression
    objective_components: dict[str, cp.Expression]
    initial_state: np.ndarray
    prices: np.ndarray
    disturbances: np.ndarray
    model: Mapping[str, Any]
    scenario_count: int
    preparation_duration_s: float
    optimization_kind: str
    # The RC model scales selected draws by the tempered-water mixing ratio.
    # Retain the original selected paths for decision-input audit only.
    source_draws_l_per_min: np.ndarray | None = None


@dataclass
class WaterHeaterPredictiveController:
    """Balance water-heating cost, comfort, and average tank temperature.

    The identified model uses delivered heat as its input, whereas OCHRE's
    public water-heater interface uses setpoint control. The first optimized
    heat decision is therefore translated to a high or low setpoint. This is
    deliberately a simple on/off wrapper around the continuous MPC problem.
    """

    model_path: Path | None = DEFAULT_MODEL_PATH
    model_kind: str = "rc" # "state-space" or "rc"
    optimization_kind: str = "milp" # "milp" or "convex"
    forecast_type: str = "perfect"
    scenario_method: str = "daily_budget_high_rf_1"
    solver: str = "GUROBI"
    solver_options: Mapping[str, Any] = field(default_factory=dict)
    #model_timestep: float = 10.0
    horizon: pd.Timedelta = pd.Timedelta(hours=12)
    model_timestep: pd.Timedelta = pd.Timedelta(minutes=15)
    equipment_name: str = "Water Heating"
    minimum_temperature_c: float = 38.0
    minimum_setpoint_c: float = 45.0
    maximum_setpoint_c: float = 60.0
    nominal_hp_capacity_kw: float | None = None
    max_cycles: int = 3
    dwell_time_off: pd.Timedelta = pd.Timedelta(minutes=15)
    dwell_time_on: pd.Timedelta = pd.Timedelta(minutes=60)
    enforce_cycle_limit: bool = True
    enforce_dwell_constraints: bool = False
    minimum_lower_temperature_c: float | None = None
    control_variation_weight: float = 0.00 # 0.003 0.002 or 0.001
    cop: float = 3.9347
    comfort_weight: float = 100.0
    legionella_temperature_c: float = (120.0 - 32.0) * 5.0 / 9.0
    legionella_weight: float = 0.001
    # Passive cooldown requires the physical RC model; state-space control
    # retains its existing draw-only indicator.
    legionella_cooldown_enabled: bool = False
    legionella_cooldown_memory: pd.Timedelta = pd.Timedelta(hours=6)
    legionella_cooldown_band_c: float = 6.0
    persistence_weight: float = 0.001
    # One-time expected terminal shortfall for MILP scenario MPC only.
    terminal_weight: float = 0.0
    # Optional soft rolling start budget, independent of the historical hard
    # horizon-wide cycle cap above. Starts are counted at interval starts.
    rolling_start_limit_24h: int | None = None
    rolling_start_slack_weight: float = 0.0
    # One-time endpoint penalty for retaining less than one available start.
    terminal_cycle_reserve_weight: float = 0.0
    water_use_peak_l_per_min: float = 0.75
    water_use_two_hour_volume_l: float = 18.0

    _model: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _last_decision_time: pd.Timestamp | None = field(
        default=None, init=False, repr=False
    )
    _last_setpoint: float | None = field(default=None, init=False, repr=False)
    _history: list[tuple[float, np.ndarray, np.ndarray]] = field(
        default_factory=list, init=False, repr=False
    )
    _last_observed_on: int | None = field(default=None, init=False, repr=False)
    _last_observed_transition_time: pd.Timestamp | None = field(
        default=None, init=False, repr=False
    )
    _recent_off_times: list[pd.Timestamp] = field(
        default_factory=list, init=False, repr=False
    )
    _observed_start_times: list[pd.Timestamp] = field(
        default_factory=list, init=False, repr=False
    )
    _last_start_observed_on: int | None = field(default=None, init=False, repr=False)
    _last_start_observed_time: pd.Timestamp | None = field(
        default=None, init=False, repr=False
    )
    _rolling_cycle_diagnostics_history: list[dict[str, Any]] = field(
        default_factory=list, init=False, repr=False
    )
    solve_durations_s: dict[str, list[float]] = field(
        default_factory=lambda: {"convex": [], "milp": []}, init=False, repr=False
    )
    _optimization_plans: list[pd.DataFrame] = field(
        default_factory=list, init=False, repr=False
    )
    _objective_component_expressions: dict[str, cp.Expression] = field(
        default_factory=dict, init=False, repr=False
    )
    _objective_component_history: list[dict[str, Any]] = field(
        default_factory=list, init=False, repr=False
    )
    _solve_history: list[dict[str, Any]] = field(
        default_factory=list, init=False, repr=False
    )
    _last_legionella_attainment_time: pd.Timestamp | None = field(
        default=None, init=False, repr=False
    )
    _legionella_mask_history: list[dict[str, Any]] = field(
        default_factory=list, init=False, repr=False
    )
    _scenario_diagnostics_history: list[dict[str, Any]] = field(
        default_factory=list, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self.horizon = pd.Timedelta(self.horizon)
        self.model_timestep = pd.Timedelta(self.model_timestep)
        if self.forecast_type not in {"perfect", "scenario"}:
            raise ValueError("forecast_type must be 'perfect' or 'scenario'")
        if self.optimization_kind not in {"convex", "milp"}:
            raise ValueError("optimization_kind must be 'convex' or 'milp'")
        if self.forecast_type == "scenario" and (
            self.horizon <= pd.Timedelta(0)
            or self.model_timestep != pd.Timedelta(minutes=15)
            or self.horizon % self.model_timestep != pd.Timedelta(0)
        ):
            raise ValueError("Scenario control requires a positive horizon in 15-minute steps")
        if not np.isfinite(self.terminal_weight) or self.terminal_weight < 0:
            raise ValueError("terminal_weight must be finite and nonnegative")
        if self.terminal_weight and (
            self.forecast_type != "scenario" or self.optimization_kind != "milp"
        ):
            raise ValueError("terminal_weight applies only to MILP scenario control")
        if self.rolling_start_limit_24h is not None and (
            isinstance(self.rolling_start_limit_24h, (bool, np.bool_))
            or not isinstance(self.rolling_start_limit_24h, (int, np.integer))
            or self.rolling_start_limit_24h < 1
        ):
            raise ValueError("rolling_start_limit_24h must be a positive integer or None")
        for name in ("rolling_start_slack_weight", "terminal_cycle_reserve_weight"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.rolling_start_limit_24h is None and (
            self.rolling_start_slack_weight or self.terminal_cycle_reserve_weight
        ):
            raise ValueError("Rolling cycle weights require rolling_start_limit_24h")
        if self.rolling_start_limit_24h is not None and self.optimization_kind != "milp":
            raise ValueError("The rolling start budget applies only to MILP control")
        self.legionella_cooldown_memory = pd.Timedelta(self.legionella_cooldown_memory)
        if (
            pd.isna(self.legionella_cooldown_memory)
            or self.legionella_cooldown_memory <= pd.Timedelta(0)
        ):
            raise ValueError("legionella_cooldown_memory must be positive")
        if (
            not np.isfinite(self.legionella_cooldown_band_c)
            or self.legionella_cooldown_band_c <= 0
        ):
            raise ValueError("legionella_cooldown_band_c must be finite and positive")

    @property
    def optimization_plans(self) -> tuple[pd.DataFrame, ...]:
        """Return the solved open-loop trajectory from every successful solve."""
        return tuple(self._optimization_plans)

    @property
    def objective_component_history(self) -> pd.DataFrame:
        """Return weighted objective terms from every successful solve."""
        return pd.DataFrame(self._objective_component_history)

    @property
    def observed_start_times(self) -> tuple[pd.Timestamp, ...]:
        """Return observed OFF-to-ON events retained from the latest 24 hours."""
        return tuple(self._observed_start_times)

    @property
    def rolling_cycle_diagnostics_history(self) -> pd.DataFrame:
        """Return per-scenario planned rolling counts and required cycle slack."""
        return pd.DataFrame(self._rolling_cycle_diagnostics_history)

    @property
    def solve_history(self) -> pd.DataFrame:
        """Return duration and status for every optimization attempt."""
        return pd.DataFrame(self._solve_history)

    @property
    def legionella_mask_history(self) -> pd.DataFrame:
        """Return precomputed draw and cooldown masks at forecast endpoints."""
        return pd.DataFrame(self._legionella_mask_history)

    @property
    def scenario_diagnostics_history(self) -> pd.DataFrame:
        """Return forecast provenance and mean first-step fixture flow."""
        return pd.DataFrame(self._scenario_diagnostics_history)

    def controls_for_step(
        self,
        timestamp: pd.Timestamp,
        previous_status: Mapping[str, Any] | None,
        current_schedule: Mapping[str, Any],
        forecast: Forecast,
    ) -> dict[str, dict[str, Any]]:
        """Return the current setpoint from a receding-horizon optimization."""
        if previous_status is None:
            return {}
        if self.forecast_type == "scenario" and self.optimization_kind == "milp":
            prepared = self.prepare_milp_scenario_step(
                timestamp, previous_status, current_schedule, forecast,
            )
            if prepared is None:
                if self._model is None:
                    return {}
                return self._setpoint_control()
            return self._solve_prepared_scenario_step(prepared)
        prior_on = self._observe_heat_pump(timestamp, previous_status)
        nominal_capacity_w = previous_status.get(
            f"{self.equipment_name} Nominal Heat Pump Capacity (W)"
        )
        if nominal_capacity_w is not None:
            self.nominal_hp_capacity_kw = float(nominal_capacity_w) / 1000
        model = self._load_model()
        if model is None:
            return {}
        measurement = self._measurement(previous_status, model["output_names"])
        if measurement is None:
            return self._setpoint_control()
        # Observe every plant step so a short attainment between MPC solves
        # is remembered. Optimized future temperatures never refresh memory.
        self._observe_legionella_temperature(timestamp, measurement)
        if (
            self._last_decision_time is not None
            and timestamp - self._last_decision_time < self.model_timestep
        ):
            return self._setpoint_control()
        if self.forecast_type == "scenario":
            # Forecast issuance follows the measured data's local quarter hours.
            if timestamp != timestamp.floor(self.model_timestep):
                return self._setpoint_control()
            return self._controls_for_scenarios(
                timestamp, previous_status, current_schedule, forecast,
                model, measurement, prior_on,
            )
        disturbances = self._forecast_disturbances(
            timestamp, current_schedule, previous_status, forecast, model
        )
        if disturbances is None:
            return self._setpoint_control()
        if self.model_kind == "rc":
            # OCHRE forecasts tempered fixture volume. Estimate the volume
            # actually withdrawn from the tank using the measured outlet proxy.
            upper = float(measurement[0])
            mains = disturbances[1]
            target = model["tempered_draw_temperature_c"]
            warm = upper > np.maximum(target, mains) + 1e-6
            ratio = np.where(
                warm,
                np.clip((target - mains) / np.maximum(upper - mains, 1e-6), 0, 1),
                1.0,
            )
            index = self._planning_index(timestamp, forecast)
            hot = np.zeros(len(index))
            for extra in ("Clothes Washer (L/min)", "Dishwasher (L/min)"):
                if extra in forecast.data:
                    values = forecast.data[extra]
                    hot += np.asarray([
                        values.loc[
                            (values.index >= start)
                            & (values.index < start + self.model_timestep)
                        ].mean()
                        for start in index
                    ])
            disturbances[0] = (disturbances[0] - hot) * ratio + hot

        delivered_heat = (
            float(previous_status.get("Water Heating Delivered (W)", 0.0)) / 1000
        )
        if self.model_kind == "rc":
            initial_state = measurement
        else:
            self._update_history(
                delivered_heat, disturbances[:, 0], measurement, model["m"]
            )
            initial_state = self._lifted_state(model["m"])
        prices = self._forecast_prices(timestamp, forecast, disturbances.shape[1])
        legionella_indicator = self._forecast_legionella_indicator(
            timestamp, forecast, disturbances.shape[1]
        )
        if prices is None or legionella_indicator is None:
            return self._setpoint_control()
        cooldown_mask = self._forecast_legionella_cooldown_mask(
            timestamp, measurement, disturbances, model
        )
        self._record_legionella_mask(
            timestamp, measurement, legionella_indicator, cooldown_mask
        )
        legionella_indicator = legionella_indicator * cooldown_mask

        if self.optimization_kind == "convex":
            problem, heat = self.optimization_problem_convex(
                initial_state,
                prices,
                disturbances,
                model,
                previous_heat_kw=delivered_heat,
                legionella_indicator=legionella_indicator,
            )
            solve_start = time.perf_counter()
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="Solution may be inaccurate.*",
                        category=UserWarning,
                    )
                    warnings.filterwarnings(
                        "ignore",
                        message=r"\s*The problem is either infeasible or unbounded.*",
                        category=UserWarning,
                    )
                    problem.solve(solver=self.solver, **dict(self.solver_options))
            except cp.error.SolverError:
                duration = time.perf_counter() - solve_start
                self.solve_durations_s["convex"].append(duration)
                self._record_solve_attempt(
                    timestamp, "convex", duration, "solver_error", False,
                    problem=problem,
                )
                return self._setpoint_control()
            duration = time.perf_counter() - solve_start
            self.solve_durations_s["convex"].append(duration)
            successful = (
                problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
                and heat.value is not None
            )
            self._record_solve_attempt(
                timestamp, "convex", duration, str(problem.status), successful,
                problem=problem, incumbent_available=heat.value is not None,
            )
            if not successful:
                return self._setpoint_control()

            self._record_objective_components(timestamp, problem)
            self._record_optimization_plan(
                timestamp, initial_state, heat.value, prices, disturbances, model
            )

            if self.persistence_weight >= 0:
                self._last_setpoint = (
                    self.maximum_setpoint_c
                    if float(heat.value[0]) > 0.01
                    else self.minimum_setpoint_c
                )
            else:
                self._last_setpoint = (
                    self.maximum_setpoint_c
                    if float(heat.value[0]) > 0.1
                    else self.minimum_setpoint_c
                )

            self._last_decision_time = timestamp
            return self._setpoint_control()


        elif self.optimization_kind == "milp":
            if prior_on is None:
                return self._setpoint_control()
            problem, heat, avg_weighted_temperature = (
                self.optimization_problem_milp(
                initial_state,
                prices,
                disturbances,
                model,
                legionella_indicator=legionella_indicator,
                prior_on=prior_on,
                recent_cycles=min(1, len(self._recent_off_times)),
                previous_heat_kw=delivered_heat,
                prediction_start=timestamp,
                prior_dwell_steps=self._remaining_dwell_steps(
                    timestamp, prior_on
                ),
            ))
            solve_start = time.perf_counter()
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="Solution may be inaccurate.*",
                        category=UserWarning,
                    )
                    warnings.filterwarnings(
                        "ignore",
                        message=r"\s*The problem is either infeasible or unbounded.*",
                        category=UserWarning,
                    )
                    problem.solve(solver=self.solver, **dict(self.solver_options))
            except cp.error.SolverError:
                duration = time.perf_counter() - solve_start
                self.solve_durations_s["milp"].append(duration)
                self._record_solve_attempt(
                    timestamp, "milp", duration, "solver_error", False,
                    problem=problem,
                )
                return self._setpoint_control()
            duration = time.perf_counter() - solve_start
            self.solve_durations_s["milp"].append(duration)
            successful = (
                problem.status
                in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE, cp.USER_LIMIT}
                and heat.value is not None
            )
            self._record_solve_attempt(
                timestamp, "milp", duration, str(problem.status), successful,
                problem=problem, incumbent_available=heat.value is not None,
            )
            if not successful:
                return self._setpoint_control()

            self._record_objective_components(timestamp, problem)
            self._record_optimization_plan(
                timestamp, initial_state, heat.value, prices, disturbances, model
            )
            self._record_rolling_cycle_diagnostics(timestamp, heat.value, prior_on)
            self._last_setpoint = (
                self.maximum_setpoint_c
                if float(heat.value[0]) > 0.1
                else self.minimum_setpoint_c
            )

            self._last_decision_time = timestamp
            return self._setpoint_control()


    def prepare_milp_scenario_step(
        self,
        timestamp: pd.Timestamp,
        previous_status: Mapping[str, Any] | None,
        current_schedule: Mapping[str, Any],
        forecast: Forecast,
    ) -> PreparedWaterHeaterScenarioStep | None:
        """Observe the plant and build the existing MILP scenario problem.

        Call this at every plant step so observed cycle and temperature memory
        remains current. It returns a problem only on an eligible MPC boundary.
        """
        if self.forecast_type != "scenario" or self.optimization_kind != "milp":
            raise ValueError("MILP scenario preparation requires MILP scenario control")
        if previous_status is None:
            return None
        prior_on = self._observe_heat_pump(timestamp, previous_status)
        nominal_capacity_w = previous_status.get(
            f"{self.equipment_name} Nominal Heat Pump Capacity (W)"
        )
        if nominal_capacity_w is not None:
            self.nominal_hp_capacity_kw = float(nominal_capacity_w) / 1000
        model = self._load_model()
        if model is None:
            return None
        measurement = self._measurement(previous_status, model["output_names"])
        if measurement is None:
            return None
        self._observe_legionella_temperature(timestamp, measurement)
        if (
            self._last_decision_time is not None
            and timestamp - self._last_decision_time < self.model_timestep
        ):
            return None
        if timestamp != timestamp.floor(self.model_timestep):
            return None
        return self._prepare_scenario_problem(
            timestamp, previous_status, current_schedule, forecast,
            model, measurement, prior_on,
        )

    def held_controls_for_step(self) -> dict[str, dict[str, Any]]:
        """Return the last water-heater command without starting another solve."""
        return self._setpoint_control()

    def held_controls(self) -> dict[str, dict[str, Any]]:
        """Compatibility alias for existing combined-controller call sites."""
        return self.held_controls_for_step()

    def _controls_for_scenarios(
        self,
        timestamp: pd.Timestamp,
        previous_status: Mapping[str, Any],
        current_schedule: Mapping[str, Any],
        forecast: Forecast,
        model: Mapping[str, Any],
        measurement: np.ndarray,
        prior_on: int | None,
    ) -> dict[str, dict[str, Any]]:
        """Solve historical water-use paths without consulting future real draws."""
        prepared = self._prepare_scenario_problem(
            timestamp, previous_status, current_schedule, forecast,
            model, measurement, prior_on,
        )
        if prepared is None:
            return self._setpoint_control()
        return self._solve_prepared_scenario_step(prepared)

    def _prepare_scenario_problem(
        self,
        timestamp: pd.Timestamp,
        previous_status: Mapping[str, Any],
        current_schedule: Mapping[str, Any],
        forecast: Forecast,
        model: Mapping[str, Any],
        measurement: np.ndarray,
        prior_on: int | None,
    ) -> PreparedWaterHeaterScenarioStep | None:
        """Prepare the unchanged single-device scenario formulation."""
        preparation_start = time.perf_counter()
        provider = forecast.water_scenarios
        if provider is None:
            raise ValueError("Scenario control requires a water-scenario forecast provider")
        if self.optimization_kind == "milp" and prior_on is None:
            return None
        scenarios = provider.scenarios_at(timestamp)
        draws = np.asarray(scenarios.values, dtype=float)
        index = self._planning_index(timestamp, forecast)
        steps = int(self.horizon / self.model_timestep)
        if (
            draws.ndim != 2 or draws.shape[0] == 0 or draws.shape[1] != steps
            or len(index) != steps or not np.isfinite(draws).all()
            or (draws < 0).any()
        ):
            raise ValueError(
                f"Scenario control requires complete nonnegative {steps}-step paths"
            )
        count = draws.shape[0]
        self._scenario_diagnostics_history.append({
            "timestamp": pd.Timestamp(timestamp),
            "scenario_method": self.scenario_method,
            "scenario_count": count,
            "optimization_probability": 1 / count,
            "first_step_mean_l_per_min": float(draws[:, 0].mean()),
            **scenarios.diagnostics,
        })

        # All uncertain draws come from the provider. Only mains temperature,
        # price, and measured ambient temperature use the shared schedule.
        rows = forecast.data.reindex(index)
        ambient = current_schedule.get(
            "Zone Temperature (C)", previous_status.get("Temperature - Indoor (C)")
        )
        values = []
        for name in model["disturbance_names"]:
            if name == "water_heater_draw_l_per_min":
                values.append(draws.copy())
            elif name == "water_heater_mains_temperature_c":
                mains = rows["Mains Temperature (C)"].to_numpy(dtype=float)
                values.append(np.broadcast_to(mains, draws.shape))
            elif name == "water_heater_tank_ambient_temperature_c":
                if ambient is None:
                    return None
                values.append(np.full(draws.shape, float(ambient)))
            else:
                raise ValueError(f"Unsupported water-heater disturbance: {name}")
        disturbances = np.stack(values, axis=1)
        if not np.isfinite(disturbances).all():
            return None
        if self.model_kind == "rc":
            mains = disturbances[:, 1, :]
            target = model["tempered_draw_temperature_c"]
            upper = float(measurement[0])
            ratio = np.where(
                upper > np.maximum(target, mains) + 1e-6,
                np.clip((target - mains) / np.maximum(upper - mains, 1e-6), 0, 1),
                1.0,
            )
            disturbances[:, 0, :] *= ratio

        delivered_heat = float(previous_status.get("Water Heating Delivered (W)", 0.0)) / 1000
        if self.model_kind == "rc":
            initial_state = measurement
        else:
            observed = disturbances[0, :, 0].copy()
            draw_position = model["disturbance_names"].index("water_heater_draw_l_per_min")
            past = forecast.data.loc[
                (forecast.data.index >= timestamp - self.model_timestep)
                & (forecast.data.index < timestamp), "Water Heating (L/min)"
            ]
            if past.empty or past.isna().any():
                return None
            observed[draw_position] = float(past.mean())
            self._update_history(delivered_heat, observed, measurement, model["m"])
            initial_state = self._lifted_state(model["m"])

        prices = self._forecast_prices(timestamp, forecast, steps)
        if prices is None:
            return None
        draw_indicators = self._scenario_legionella_indicators(timestamp, forecast, draws)
        cooldown = self._forecast_legionella_cooldown_mask(
            timestamp, measurement, disturbances[0], model
        )
        # The mean diagnostic is the fraction of scenarios with the penalty enabled.
        self._record_legionella_mask(timestamp, measurement, draw_indicators.mean(axis=0), cooldown)
        indicators = draw_indicators * cooldown
        options = dict(previous_heat_kw=delivered_heat, legionella_indicator=indicators)
        if self.optimization_kind == "convex":
            problem, heat = self.optimization_problem_convex_scenario(
                initial_state, prices, disturbances, model=model, **options
            )
        else:
            problem, heat, _ = self.optimization_problem_milp_scenario(
                initial_state, prices, disturbances, model,
                prior_on=prior_on, recent_cycles=min(1, len(self._recent_off_times)),
                prediction_start=timestamp,
                prior_dwell_steps=self._remaining_dwell_steps(timestamp, prior_on),
                **options,
            )
        cop = model["rc"].cop if self.model_kind == "rc" else self.cop
        return PreparedWaterHeaterScenarioStep(
            timestamp=pd.Timestamp(timestamp), problem=problem, heat=heat,
            electric_power_kw=heat / cop,
            objective_components=dict(self._objective_component_expressions),
            initial_state=initial_state, prices=prices,
            disturbances=disturbances, model=model, scenario_count=count,
            preparation_duration_s=time.perf_counter() - preparation_start,
            optimization_kind=self.optimization_kind,
            source_draws_l_per_min=draws.copy(),
        )

    def _solve_prepared_scenario_step(
        self, prepared: PreparedWaterHeaterScenarioStep,
    ) -> dict[str, dict[str, Any]]:
        """Solve one prepared scenario problem in standalone operation."""
        timestamp = prepared.timestamp
        problem = prepared.problem
        heat = prepared.heat
        start = time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Solution may be inaccurate.*")
                warnings.filterwarnings("ignore", message=r"\s*The problem is either infeasible or unbounded.*")
                problem.solve(solver=self.solver, **dict(self.solver_options))
            status = str(problem.status)
            accepted = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
            if prepared.optimization_kind == "milp":
                accepted.add(cp.USER_LIMIT)
            successful = problem.status in accepted and heat.value is not None
        except cp.error.SolverError:
            status, successful = "solver_error", False
        duration = time.perf_counter() - start
        self.solve_durations_s[prepared.optimization_kind].append(duration)
        self._record_solve_attempt(
            timestamp, prepared.optimization_kind, duration, status, successful,
            problem=problem, incumbent_available=heat.value is not None,
        )
        self._solve_history[-1].update(
            forecast_type="scenario", scenario_count=prepared.scenario_count,
            preparation_duration_s=prepared.preparation_duration_s,
        )
        if not successful:
            return self._setpoint_control()
        if prepared.optimization_kind == "milp":
            return self.apply_milp_scenario_solution(prepared)
        self._record_objective_components(
            timestamp, problem, prepared.objective_components,
        )
        self._record_scenario_optimization_plan(
            timestamp, prepared.initial_state, heat.value, prepared.prices,
            prepared.disturbances, prepared.model,
        )
        threshold = 0.01 if self.persistence_weight >= 0 else 0.1
        self._last_setpoint = (
            self.maximum_setpoint_c if float(heat.value[0, 0]) > threshold
            else self.minimum_setpoint_c
        )
        self._last_decision_time = timestamp
        return self._setpoint_control()

    def apply_milp_scenario_solution(
        self, prepared: PreparedWaterHeaterScenarioStep,
    ) -> dict[str, dict[str, Any]]:
        """Record a solved MILP scenario plan and apply its shared first heat."""
        if prepared.optimization_kind != "milp":
            raise ValueError("Expected a prepared MILP scenario problem")
        heat_value = prepared.heat.value
        if heat_value is None:
            raise ValueError("Prepared MILP scenario problem has no heat solution")
        self._record_objective_components(
            prepared.timestamp, prepared.problem, prepared.objective_components,
        )
        self._record_scenario_optimization_plan(
            prepared.timestamp, prepared.initial_state, heat_value,
            prepared.prices, prepared.disturbances, prepared.model,
        )
        self._record_rolling_cycle_diagnostics(
            prepared.timestamp, heat_value, self._last_observed_on,
        )
        self._last_setpoint = (
            self.maximum_setpoint_c if float(heat_value[0, 0]) > 0.1
            else self.minimum_setpoint_c
        )
        self._last_decision_time = prepared.timestamp
        return self._setpoint_control()

    def _scenario_legionella_indicators(
        self, timestamp: pd.Timestamp, forecast: Forecast, draws: np.ndarray,
    ) -> np.ndarray:
        """Combine measured history with each 15-minute forecast path.

        Within each future interval the predicted mean flow is held constant;
        future one-minute peaks in the actual test profile remain unobserved.
        """
        observed = forecast.data.loc[
            (forecast.data.index >= timestamp - pd.Timedelta(hours=2))
            & (forecast.data.index < timestamp), "Water Heating (L/min)"
        ].resample("min").mean().ffill()
        minutes_per_step = int(self.model_timestep / pd.Timedelta(minutes=1))
        future_index = pd.date_range(timestamp, periods=draws.shape[1] * minutes_per_step, freq="min")
        indicators = []
        for path in draws:
            future = pd.Series(np.repeat(path, minutes_per_step), index=future_index)
            source = Forecast(pd.concat([observed, future]).to_frame("Water Heating (L/min)"))
            indicator = self._forecast_legionella_indicator(timestamp, source, draws.shape[1])
            if indicator is None:
                raise ValueError("Scenario water-use history is incomplete")
            indicators.append(indicator)
        return np.asarray(indicators)

    def _record_scenario_optimization_plan(
        self, timestamp: pd.Timestamp, initial_state: np.ndarray,
        heat: np.ndarray, prices: np.ndarray, disturbances: np.ndarray,
        model: Mapping[str, Any],
    ) -> None:
        """Average solved trajectories after applying each path's own dynamics."""
        paths = []
        for scenario in range(len(heat)):
            self._record_optimization_plan(
                timestamp, initial_state, heat[scenario], prices, disturbances[scenario], model
            )
            paths.append(self._optimization_plans.pop())
        values = np.stack([path.to_numpy(dtype=float) for path in paths])
        plan = pd.DataFrame(values.mean(axis=0), index=paths[0].index, columns=paths[0].columns)
        for name in model["output_names"]:
            position = plan.columns.get_loc(name)
            plan[f"{name}_p10"] = np.quantile(values[:, :, position], 0.1, axis=0)
            plan[f"{name}_p90"] = np.quantile(values[:, :, position], 0.9, axis=0)
        plan.attrs.update(solved_at=pd.Timestamp(timestamp), scenario_count=len(paths))
        self._optimization_plans.append(plan)

    def optimization_problem_convex(
        self,
        initial_state: np.ndarray,
        prices: np.ndarray,
        disturbances: np.ndarray,
        model: Mapping[str, Any] | None = None,
        *,
        previous_heat_kw: float,
        legionella_indicator: np.ndarray | None = None,
    ) -> tuple[cp.Problem, cp.Variable]:
        """Build the state-space MPC problem and return it with its heat input.

        Penalize electricity use, low upper-tank temperature, and low average
        tank temperature when the forecast has no significant recent draw.
        """
        model = model or self._load_model()
        if model is None:
            raise FileNotFoundError("Water-heater state-space model is unavailable")

        initial_state = np.asarray(initial_state, dtype=float).reshape(-1)
        prices = np.asarray(prices, dtype=float).reshape(-1)
        disturbances = np.asarray(disturbances, dtype=float)
        previous_heat_kw = float(previous_heat_kw)
        if not np.isfinite(previous_heat_kw) or previous_heat_kw < 0:
            raise ValueError("previous_heat_kw must be finite and nonnegative")
        if disturbances.ndim != 2:
            raise ValueError(
                "disturbances must have shape (n_disturbances, horizon)"
            )
        steps = disturbances.shape[1]
        if legionella_indicator is None:
            legionella_indicator = np.ones(steps)
        else:
            legionella_indicator = np.asarray(
                legionella_indicator, dtype=float
            ).reshape(-1)
        if (
            steps == 0
            or prices.size != steps
            or legionella_indicator.size != steps
            or not np.isin(legionella_indicator, (0.0, 1.0)).all()
            or initial_state.size != model["A"].shape[0]
            or disturbances.shape[0] != model["E"].shape[1]
        ):
            raise ValueError(
                "State, price, indicator, or disturbance dimensions do not match the model"
            )

        state = cp.Variable((model["A"].shape[0], steps + 1))
        heat = cp.Variable(steps, nonneg=True)
        output_temperatures = model["output_matrix"] @ state[:, 1:]
        upper_temperature = output_temperatures[0:1, :]
        lower_temperature = output_temperatures[1:2, :]
        heat_limit = self._heat_limit_kw(model)
        average_temperature = (upper_temperature + lower_temperature) / 2
        avg_weighted_temperature = (0.75 * upper_temperature +
                                    0.25 * lower_temperature)

        constraints = [state[:, 0] == initial_state, heat <= heat_limit]#,
                       #avg_weighted_temperature <= self.maximum_setpoint_c]
        for step in range(steps):
            if self.model_kind == "rc":
                a, b, w = rc_step_matrices(
                    model["rc"], *disturbances[:, step],
                    self.model_timestep.total_seconds() / 3600,
                )
                next_state = a @ state[:, step] + b * heat[step] + w
            else:
                next_state = (
                    model["A"] @ state[:, step]
                    + model["B"][:, 0] * heat[step]
                    + model["E"] @ disturbances[:, step]
                )
            constraints.append(state[:, step + 1] == next_state)

        timestep_hours = self.model_timestep.total_seconds() / 3600
        electricity_cost = 1/steps * cp.sum(
            cp.multiply(prices, heat / (model["rc"].cop if self.model_kind == "rc" else self.cop) * timestep_hours)
        )
        comfort_cost =  (self.comfort_weight * 1/steps * timestep_hours
                         * cp.sum(cp.pos(
                         self.minimum_temperature_c - upper_temperature)
        ))
        legionella_cost = (self.legionella_weight * 1/steps *
                           timestep_hours * cp.sum(
            cp.multiply(
                legionella_indicator.reshape(1, -1),
                cp.pos(self.legionella_temperature_c - avg_weighted_temperature),
            )
        ))

        heat_variation = cp.abs(heat[0] - previous_heat_kw)
        if steps > 1:
            heat_variation += cp.norm1(heat[1:] - heat[:-1])
        persistence_cost = (self.persistence_weight * 1/steps * timestep_hours
                            * heat_variation)

        self._objective_component_expressions = {
            "electricity_objective_cost": electricity_cost,
            "comfort_objective_cost": comfort_cost,
            "legionella_objective_cost": legionella_cost,
            "persistence_objective_cost": persistence_cost,
        }
        return (
            cp.Problem(
                cp.Minimize(electricity_cost + comfort_cost +
                            legionella_cost + persistence_cost),
                constraints,
            ),
            heat,
        )

    def optimization_problem_convex_scenario(
        self,
        initial_state: np.ndarray,
        prices: np.ndarray,
        disturbances: np.ndarray,
        num_scenarios: int | None = None,
        model: Mapping[str, Any] | None = None,
        *,
        previous_heat_kw: float,
        legionella_indicator: np.ndarray | None = None,
    ) -> tuple[cp.Problem, cp.Expression]:
        """Build equally probable scenarios with a shared first heat decision."""
        problem, heat, _ = build_scenario_problem(
            self, "convex", initial_state, prices, disturbances, model,
            num_scenarios=num_scenarios,
            previous_heat_kw=previous_heat_kw,
            legionella_indicator=legionella_indicator,
        )
        return problem, heat

    def _record_objective_components(
        self,
        timestamp: pd.Timestamp,
        problem: cp.Problem,
        expressions: Mapping[str, cp.Expression] | None = None,
    ) -> None:
        """Record the weighted terms making up one solved MPC objective."""
        components = (
            self._objective_component_expressions
            if expressions is None else expressions
        )
        if not components:
            return
        values = {
            name: float(np.asarray(expression.value).squeeze())
            for name, expression in components.items()
        }
        if not all(np.isfinite(value) for value in values.values()):
            raise RuntimeError("Solved convex objective contains a non-finite term")
        component_sum = float(sum(values.values()))
        # A joint HEMS solve uses these expressions without solving this child
        # cp.Problem separately. Its own objective is still the component sum.
        scalar_objective = (
            component_sum if problem.value is None else float(problem.value)
        )
        self._objective_component_history.append(
            {
                "timestamp": pd.Timestamp(timestamp),
                **values,
                "scalar_objective_cost": scalar_objective,
                "component_sum": component_sum,
                "objective_balance_error": scalar_objective - component_sum,
            }
        )

    def _record_solve_attempt(
        self,
        timestamp: pd.Timestamp,
        optimization_kind: str,
        duration_s: float,
        status: str,
        successful: bool,
        *,
        problem: cp.Problem | None = None,
        incumbent_available: bool | None = None,
    ) -> None:
        """Record one solver attempt, including failures that trigger fallback."""
        solver_stats = getattr(problem, "solver_stats", None)
        solver_time = None if solver_stats is None else solver_stats.solve_time
        raw_status = None
        if solver_stats is not None and str(self.solver).upper() == "GUROBI":
            gurobi_model = solver_stats.extra_stats
            if gurobi_model is not None:
                raw_status = int(gurobi_model.Status)
        if incumbent_available is None:
            incumbent_available = successful
        self._solve_history.append(
            {
                "timestamp": pd.Timestamp(timestamp),
                "optimization_kind": optimization_kind,
                "duration_s": float(duration_s),
                "solver_time_s": None if solver_time is None else float(solver_time),
                "status": status,
                "solver_status_code": raw_status,
                # Gurobi status 9 is TIME_LIMIT, even when CVXPY relabels a
                # no-incumbent solve as infeasible_inaccurate.
                "time_limit_hit": raw_status == 9,
                "incumbent_available": bool(incumbent_available),
                "successful": bool(successful),
                "solve_failed": not bool(successful),
            }
        )


    def optimization_problem_milp(
        self,
        initial_state: np.ndarray,
        prices: np.ndarray,
        disturbances: np.ndarray,
        model: Mapping[str, Any] | None = None,
        *,
        legionella_indicator: np.ndarray | None = None,
        prior_on: int,
        recent_cycles: int,
        previous_heat_kw: float,
        prior_dwell_steps: int = 0,
        prediction_start: pd.Timestamp | None = None,
    ) -> tuple[cp.Problem, cp.Variable, cp.Expression]:
        """Build the state-space MPC problem and return it with its heat input.

        Penalize electricity use, low upper-tank temperature, low average
        tank temperature outside substantial draws, and changes in heat input.
        """
        model = model or self._load_model()
        if model is None:
            raise FileNotFoundError("Water-heater state-space model is unavailable")

        initial_state = np.asarray(initial_state, dtype=float).reshape(-1)
        prices = np.asarray(prices, dtype=float).reshape(-1)
        disturbances = np.asarray(disturbances, dtype=float)
        previous_heat_kw = float(previous_heat_kw)
        if not np.isfinite(previous_heat_kw) or previous_heat_kw < 0:
            raise ValueError("previous_heat_kw must be finite and nonnegative")
        if disturbances.ndim != 2:
            raise ValueError(
                "disturbances must have shape (n_disturbances, horizon)"
            )
        steps = disturbances.shape[1]
        if legionella_indicator is None:
            legionella_indicator = np.ones(steps)
        else:
            legionella_indicator = np.asarray(
                legionella_indicator, dtype=float
            ).reshape(-1)
        if (
                steps == 0
                or prices.size != steps
                or legionella_indicator.size != steps
                or not np.isin(legionella_indicator, (0.0, 1.0)).all()
                or initial_state.size != model["A"].shape[0]
                or disturbances.shape[0] != model["E"].shape[1]
        ):
            raise ValueError(
                "State, price, indicator, or disturbance dimensions do not match the model"
            )

        state = cp.Variable((model["A"].shape[0], steps + 1))
        heat = cp.Variable(steps, nonneg=True)
        heat_state = cp.Variable(steps, boolean=True)
        output_temperatures = model["output_matrix"] @ state[:, 1:]
        upper_temperature = output_temperatures[0:1, :]
        lower_temperature = output_temperatures[1:2, :]
        heat_limit = self._heat_limit_kw(model)
        average_temperature = (upper_temperature + lower_temperature) / 2
        avg_weighted_temperature = (0.75*upper_temperature +
                                    0.25*lower_temperature)
        constraints = [
            state[:, 0] == initial_state,
            heat == heat_limit*heat_state,
            avg_weighted_temperature <= self.maximum_setpoint_c
        ]
        track_transitions = (
            self.enforce_cycle_limit or self.enforce_dwell_constraints
            or self.rolling_start_limit_24h is not None
        )
        heat_on = heat_off = None
        if track_transitions:
            heat_off = cp.Variable(steps, boolean=True)
            heat_on = cp.Variable(steps, boolean=True)
            previous_heat_state = cp.hstack([prior_on, heat_state[:-1]])
            constraints.extend([
                heat_state - previous_heat_state == heat_on - heat_off,
                heat_on + heat_off <= 1,
            ])
            if self.enforce_cycle_limit:
                constraints.append(cp.sum(heat_on) + recent_cycles <= self.max_cycles)
        if self.minimum_lower_temperature_c is not None:
            constraints.append(
                lower_temperature >= self.minimum_lower_temperature_c
            )
        if self.enforce_dwell_constraints and prior_dwell_steps:
            constraints.append(
                heat_state[:min(prior_dwell_steps, steps)] == prior_on
            )
        for step in range(steps):
            if self.model_kind == "rc":
                a, b, w = rc_step_matrices(
                    model["rc"], *disturbances[:, step],
                    self.model_timestep.total_seconds() / 3600,
                )
                next_state = a @ state[:, step] + b * heat[step] + w
            else:
                next_state = (
                        model["A"] @ state[:, step]
                        + model["B"][:, 0] * heat[step]
                        + model["E"] @ disturbances[:, step]
                )
            constraints.append(state[:, step + 1] == next_state)

        if self.enforce_dwell_constraints:
            assert heat_on is not None and heat_off is not None
            dwell_steps_on = int(np.ceil(self.dwell_time_on / self.model_timestep))
            dwell_steps_off = int(np.ceil(self.dwell_time_off / self.model_timestep))
            for step in range(steps):
                on_window = min(dwell_steps_on, steps - step)
                off_window = min(dwell_steps_off, steps - step)
                constraints.extend([
                    cp.sum(heat_state[step:step + on_window])
                    >= on_window * heat_on[step],
                    cp.sum(1 - heat_state[step:step + off_window])
                    >= off_window * heat_off[step],
                ])
        timestep_hours = self.model_timestep.total_seconds() / 3600

        cycle_components: dict[str, cp.Expression] = {}
        if self.rolling_start_limit_24h is not None:
            if prediction_start is None or pd.isna(pd.Timestamp(prediction_start)):
                raise ValueError("The rolling start budget requires prediction_start")
            assert heat_on is not None
            starts_at = pd.date_range(
                pd.Timestamp(prediction_start), periods=steps, freq=self.model_timestep,
            )
            slack = cp.Variable(steps, nonneg=True, name="water_heater_cycle_slack")
            for step, time_at in enumerate(starts_at):
                left = time_at - pd.Timedelta(hours=24)
                observed = sum(left < event <= starts_at[0] for event in self._observed_start_times)
                retained = np.flatnonzero((starts_at > left) & (starts_at <= time_at))
                constraints.append(
                    observed + cp.sum(heat_on[retained])
                    <= self.rolling_start_limit_24h + slack[step]
                )
            cycle_components["cycle_slack_objective_cost"] = (
                self.rolling_start_slack_weight * timestep_hours / steps * cp.sum(slack)
            )
            endpoint = starts_at[0] + steps * self.model_timestep
            left = endpoint - pd.Timedelta(hours=24)
            observed = sum(left < event <= starts_at[0] for event in self._observed_start_times)
            retained = np.flatnonzero(starts_at > left)
            endpoint_starts = observed + cp.sum(heat_on[retained])
            cycle_components["terminal_cycle_reserve_objective_cost"] = (
                self.terminal_cycle_reserve_weight
                * cp.pos(endpoint_starts - (self.rolling_start_limit_24h - 1))
            )

        if self.persistence_weight:
            heat_variation = cp.abs(heat[0] - previous_heat_kw)
            if steps > 1:
                heat_variation += cp.norm1(heat[1:] - heat[:-1])
            persistence_cost = (
                self.persistence_weight / steps * timestep_hours * heat_variation
            )
        else:
            persistence_cost = cp.Constant(0.0)

        electricity_cost = 1/steps * cp.sum(
            cp.multiply(prices, heat / (model[
                                            "rc"].cop if self.model_kind == "rc" else self.cop) * timestep_hours)
        )
        comfort_cost = self.comfort_weight * 1/steps * timestep_hours * cp.sum(
            cp.pos(self.minimum_temperature_c - upper_temperature)
        )

        legionella_cost = self.legionella_weight * 1/steps * timestep_hours * cp.sum(
            cp.multiply(
                legionella_indicator.reshape(1, -1),
                cp.pos(self.legionella_temperature_c - avg_weighted_temperature),
            )
        )
        self._objective_component_expressions = {
            "electricity_objective_cost": electricity_cost,
            "comfort_objective_cost": comfort_cost,
            "legionella_objective_cost": legionella_cost,
            "persistence_objective_cost": persistence_cost,
            **cycle_components,
        }
        return (
            cp.Problem(
                cp.Minimize(
                    electricity_cost + comfort_cost + legionella_cost
                    + persistence_cost + sum(cycle_components.values())
                ),
                constraints,
            ),
            heat, avg_weighted_temperature
        )


    def optimization_problem_milp_scenario(
        self,
        initial_state: np.ndarray,
        prices: np.ndarray,
        disturbances: np.ndarray,
        model: Mapping[str, Any] | None = None,
        *,
        legionella_indicator: np.ndarray | None = None,
        num_scenarios: int | None = None,
        prior_on: int,
        recent_cycles: int,
        previous_heat_kw: float,
        prior_dwell_steps: int = 0,
        prediction_start: pd.Timestamp | None = None,
    ) -> tuple[cp.Problem, cp.Expression, cp.Expression]:
        """Apply the deterministic MILP constraints to every water-use path."""
        problem, heat, temperatures = build_scenario_problem(
            self, "milp", initial_state, prices, disturbances, model,
            num_scenarios=num_scenarios,
            previous_heat_kw=previous_heat_kw,
            legionella_indicator=legionella_indicator,
            prior_on=prior_on,
            recent_cycles=recent_cycles,
            prior_dwell_steps=prior_dwell_steps,
            prediction_start=prediction_start,
        )
        assert temperatures is not None
        return problem, heat, temperatures

    def _record_optimization_plan(
        self,
        timestamp: pd.Timestamp,
        initial_state: np.ndarray,
        heat: np.ndarray,
        prices: np.ndarray,
        disturbances: np.ndarray,
        model: Mapping[str, Any],
    ) -> None:
        """Reconstruct and retain one complete solved open-loop plan."""
        heat = np.maximum(np.asarray(heat, dtype=float).reshape(-1), 0)
        prices = np.asarray(prices, dtype=float).reshape(-1)
        disturbances = np.asarray(disturbances, dtype=float)
        steps = heat.size
        if steps == 0 or prices.size != steps or disturbances.shape[1] != steps:
            raise ValueError("Solved water-heater plan dimensions do not match")

        states = [np.asarray(initial_state, dtype=float).reshape(-1)]
        for step in range(steps):
            if self.model_kind == "rc":
                transition, gain, forcing = rc_step_matrices(
                    model["rc"],
                    *disturbances[:, step],
                    self.model_timestep.total_seconds() / 3600,
                )
                following = (
                    transition @ states[-1] + gain * heat[step] + forcing
                )
            else:
                following = (
                    model["A"] @ states[-1]
                    + model["B"][:, 0] * heat[step]
                    + model["E"] @ disturbances[:, step]
                )
            states.append(np.asarray(following, dtype=float).reshape(-1))

        output_temperatures = model["output_matrix"] @ np.column_stack(states)
        index = pd.date_range(
            pd.Timestamp(timestamp), periods=steps + 1, freq=self.model_timestep
        )
        # Repeat interval quantities at the final boundary so post-step plots
        # display the complete last optimization interval.
        planned_heat = np.append(heat, heat[-1])
        planned_prices = np.append(prices, prices[-1])
        planned_setpoints = np.where(
            planned_heat > 0.1,
            self.maximum_setpoint_c,
            self.minimum_setpoint_c,
        )
        cop = model["rc"].cop if self.model_kind == "rc" else self.cop
        disturbance_names = tuple(model.get("disturbance_names", ()))
        draw_name = "water_heater_draw_l_per_min"
        if draw_name in disturbance_names:
            draw = disturbances[disturbance_names.index(draw_name)]
            planned_draw = np.append(draw, draw[-1])
        else:
            planned_draw = np.full(steps + 1, np.nan)

        plan = pd.DataFrame(index=index)
        plan.index.name = "timestamp"
        for row, name in enumerate(model["output_names"]):
            plan[name] = output_temperatures[row]
        plan["water_heater_setpoint_temperature_c"] = planned_setpoints
        plan["water_heater_delivered_heat_kw"] = planned_heat
        plan["water_heater_electric_power_kw"] = planned_heat / cop
        plan["electricity_price_usd_per_kwh"] = planned_prices
        plan["hot_water_flow_rate_l_per_min"] = planned_draw
        plan.attrs["solved_at"] = pd.Timestamp(timestamp)
        self._optimization_plans.append(plan)

    def _heat_limit_kw(self, model: Mapping[str, Any]) -> float:
        if self.model_kind == "rc":
            return model["rc"].maximum_heat_kw
        capacity = self.nominal_hp_capacity_kw
        if capacity is None or not np.isfinite(capacity) or capacity <= 0:
            raise ValueError("State-space control requires a positive nominal heat-pump capacity")
        return capacity

    def _observed_heat_pump_on(self, previous_status: Mapping[str, Any]) -> int | None:
        mode = previous_status.get(f"{self.equipment_name} Mode")
        if mode == "Heat Pump On":
            return 1
        if mode in {"Off", "Upper On", "Lower On"}:
            return 0
        on_fraction = previous_status.get(f"{self.equipment_name} Heat Pump On Fraction (-)")
        if on_fraction is None or not np.isfinite(on_fraction):
            return None
        return int(float(on_fraction) > 0)

    def observe_start_history(
        self, timestamp: pd.Timestamp, previous_status: Mapping[str, Any],
    ) -> int | None:
        """Observe a completed plant sample without building or solving an MPC.

        Use during native warm-up as well as normal operation. OCHRE's ``Time``
        identifies the completed sample's interval start; a controller issue
        may occur one plant step later. The first observed ON state establishes
        state only, because its start time is unknown.
        """
        on = self._observed_heat_pump_on(previous_status)
        if on is None:
            return None
        try:
            observed_at = pd.Timestamp(previous_status.get("Time"))
        except (TypeError, ValueError):
            observed_at = pd.NaT
        if pd.isna(observed_at):
            observed_at = pd.Timestamp(timestamp)
        if pd.isna(observed_at):
            raise ValueError("A water-heater observation requires a valid timestamp")
        if (
            self._last_start_observed_time is not None
            and observed_at <= self._last_start_observed_time
        ):
            return self._last_start_observed_on
        if self._last_start_observed_on == 0 and on == 1:
            self._observed_start_times.append(observed_at)
        cutoff = observed_at - pd.Timedelta(hours=24)
        self._observed_start_times = [
            event for event in self._observed_start_times if event > cutoff
        ]
        self._last_start_observed_on = on
        self._last_start_observed_time = observed_at
        return on

    def _record_rolling_cycle_diagnostics(
        self, timestamp: pd.Timestamp, heat: np.ndarray, prior_on: int | None,
    ) -> None:
        """Save path-specific counts separately from the mean temperature plan."""
        if self.rolling_start_limit_24h is None:
            return
        heat = np.atleast_2d(np.asarray(heat, dtype=float))
        if prior_on is None:
            raise ValueError("Rolling cycle diagnostics require the measured heat-pump state")
        steps = heat.shape[1]
        times = pd.date_range(pd.Timestamp(timestamp), periods=steps, freq=self.model_timestep)
        endpoint = pd.Timestamp(timestamp) + steps * self.model_timestep
        for scenario, path in enumerate(heat):
            on = path > 1e-6
            starts = on & ~np.r_[bool(prior_on), on[:-1]]
            for step, time_at in enumerate([*times, endpoint]):
                left = time_at - pd.Timedelta(hours=24)
                observed = sum(
                    left < event <= times[0] for event in self._observed_start_times
                )
                predicted = int(starts[(times > left) & (times <= time_at)].sum())
                count = observed + predicted
                self._rolling_cycle_diagnostics_history.append({
                    "solved_at": pd.Timestamp(timestamp), "scenario": scenario,
                    "timestamp": time_at, "step": step, "is_endpoint": step == steps,
                    "predicted_start": bool(starts[step]) if step < steps else False,
                    "observed_starts_24h": observed,
                    "predicted_starts_24h": predicted,
                    "rolling_starts_24h": count,
                    "minimum_cycle_slack_starts": max(0, count - self.rolling_start_limit_24h),
                    "terminal_reserve_shortfall_starts": (
                        max(0, count - (self.rolling_start_limit_24h - 1))
                        if step == steps else 0
                    ),
                })

    def _observe_heat_pump(
        self, timestamp: pd.Timestamp, previous_status: Mapping[str, Any]
    ) -> int | None:
        """Track observed heat-pump stops over the preceding hour."""
        self.observe_start_history(timestamp, previous_status)
        cutoff = timestamp - pd.Timedelta(hours=1)
        self._recent_off_times = [
            stopped_at for stopped_at in self._recent_off_times if stopped_at > cutoff
        ]

        mode = previous_status.get(f"{self.equipment_name} Mode")
        if mode == "Heat Pump On":
            on = 1
        elif mode in {"Off", "Upper On", "Lower On"}:
            on = 0
        else:
            on_fraction = previous_status.get(
                f"{self.equipment_name} Heat Pump On Fraction (-)"
            )
            if on_fraction is None or not np.isfinite(on_fraction):
                return None
            on = int(float(on_fraction) > 0)

        if self._last_observed_on is not None and self._last_observed_on != on:
            self._last_observed_transition_time = timestamp
            if self._last_observed_on == 1:
                self._recent_off_times.append(timestamp)
        self._last_observed_on = on
        return on

    def _remaining_dwell_steps(self, timestamp: pd.Timestamp, on: int) -> int:
        """Return model intervals still required in the observed mode."""
        if self._last_observed_transition_time is None:
            return 0
        required = self.dwell_time_on if on else self.dwell_time_off
        remaining = required - (timestamp - self._last_observed_transition_time)
        if remaining <= pd.Timedelta(0):
            return 0
        return int(np.ceil(remaining / self.model_timestep))

    def _load_model(self) -> dict[str, Any] | None:
        if self._model is not None:
            return self._model
        if self.model_kind not in {"state-space", "rc"}:
            raise ValueError(f"Unknown water-heater model kind: {self.model_kind}")
        if self.model_path is None:
            return None
        model_path = Path(self.model_path)
        metadata_path = model_path / "metadata.json"
        if not metadata_path.is_file():
            return None

        with metadata_path.open(encoding="utf-8") as stream:
            metadata = json.load(stream)
        if self.model_kind == "rc":
            if metadata.get("model_type") != "two_node_water_heater_rc":
                raise ValueError("Selected RC artifact has the wrong model type")
            rc = WaterHeaterRCModel(**metadata["parameters"])
            self._model = {
                "rc": rc,
                "A": np.eye(2),
                "E": np.zeros((2, 3)),
                "output_matrix": np.eye(2),
                "output_names": tuple(metadata["columns"]["outputs"]),
                "disturbance_names": tuple(metadata["columns"]["disturbances"]),
                "tempered_draw_temperature_c": float(metadata["tempered_draw_temperature_c"]),
            }
            return self._model
        arrays_path = model_path / "model_arrays.npz"
        if not arrays_path.is_file():
            return None
        with np.load(arrays_path) as arrays:
            matrices = {name: arrays[name] for name in ("A", "B", "E")}

        output_names = tuple(metadata["columns"]["outputs"])
        if metadata["columns"]["input"] != ["water_heater_delivered_heat_kw"]:
            # A legacy W-input model cannot be used with the kW controller.
            # Leave native OCHRE control in place until the model is retrained.
            return None
        # Regularized fits retain all disturbance channels in A and E; inactive
        # channels simply have zero fitted coefficients.
        disturbance_names = tuple(metadata["columns"]["disturbances"])
        if (
            matrices["B"].shape[1] != 1
            or matrices["E"].shape[1] != len(disturbance_names)
            or matrices["A"].shape[0] != matrices["A"].shape[1]
            or matrices["A"].shape[0]
            != int(metadata["selection"]["memory"])
            * (1 + len(disturbance_names) + len(output_names))
        ):
            raise ValueError("Water-heater model matrices do not match its metadata")

        if len(output_names) != 2:
            raise ValueError("Water-heater model requires upper and lower sensors")
        output_matrix = np.zeros((2, matrices["A"].shape[0]))
        output_matrix[:, -2:] = np.eye(2)
        self._model = {
            **matrices,
            "m": int(metadata["selection"]["memory"]),
            "output_names": output_names,
            "disturbance_names": disturbance_names,
            "output_matrix": output_matrix,
        }
        return self._model

    def _forecast_disturbances(
        self,
        timestamp: pd.Timestamp,
        current_schedule: Mapping[str, Any],
        previous_status: Mapping[str, Any],
        forecast: Forecast,
        model: Mapping[str, Any],
    ) -> np.ndarray | None:
        index = self._planning_index(timestamp, forecast)
        if index.empty:
            return None
        rows = forecast.data.reindex(index)
        if rows.isnull().all(axis=None):
            return None

        values = []
        for name in model["disturbance_names"]:
            if name == "water_heater_draw_l_per_min":
                draws = forecast.data["Water Heating (L/min)"].copy()
                if self.model_kind == "rc":
                    for extra in ("Clothes Washer (L/min)", "Dishwasher (L/min)"):
                        if extra in forecast.data:
                            draws = draws.add(forecast.data[extra], fill_value=0)
                series = pd.Series(
                    [
                        draws.loc[
                            (draws.index >= start)
                            & (draws.index < start + self.model_timestep)
                        ].mean()
                        for start in index
                    ],
                    index=index,
                )
            elif name == "water_heater_mains_temperature_c":
                series = rows["Mains Temperature (C)"]
            elif name == "water_heater_tank_ambient_temperature_c":
                ambient = current_schedule.get(
                    "Zone Temperature (C)",
                    previous_status.get("Temperature - Indoor (C)"),
                )
                if ambient is None:
                    return None
                series = pd.Series(float(ambient), index=index)
            else:
                raise ValueError(f"Unsupported water-heater disturbance: {name}")
            if series.isnull().any():
                return None
            values.append(series.to_numpy(dtype=float))
        return np.vstack(values)

    def _forecast_prices(
        self,
        timestamp: pd.Timestamp,
        forecast: Forecast,
        steps: int,
    ) -> np.ndarray | None:
        index = self._planning_index(timestamp, forecast)[:steps]
        prices = forecast.data.reindex(index)[PRICE_COLUMN]
        if len(prices) != steps or prices.isnull().any():
            return None
        return prices.to_numpy(dtype=float)

    def _observe_legionella_temperature(
        self, timestamp: pd.Timestamp, measurement: np.ndarray
    ) -> None:
        """Remember measured target attainment for at most six hours."""
        if not self.legionella_cooldown_enabled or self.model_kind != "rc":
            return
        temperatures = np.asarray(measurement, dtype=float).reshape(-1)
        if temperatures.size != 2 or not np.isfinite(temperatures).all():
            return
        weighted = float(np.array([0.75, 0.25]) @ temperatures)
        if weighted >= self.legionella_temperature_c:
            self._last_legionella_attainment_time = pd.Timestamp(timestamp)
        elif (
            weighted < self.legionella_temperature_c - self.legionella_cooldown_band_c
            or (
                self._last_legionella_attainment_time is not None
                and timestamp - self._last_legionella_attainment_time
                >= self.legionella_cooldown_memory
            )
        ):
            self._last_legionella_attainment_time = None

    def _forecast_legionella_cooldown_mask(
        self,
        timestamp: pd.Timestamp,
        measurement: np.ndarray,
        disturbances: np.ndarray,
        model: Mapping[str, Any],
    ) -> np.ndarray:
        """Mask the penalty during a recent attainment's passive RC cooldown.

        Forecast only ambient losses and internal conduction, with zero draws
        and zero heating. The mask is fixed before either optimization is
        built and expires at the first below-band endpoint or memory expiry.
        """
        mask = np.ones(disturbances.shape[1])
        if (
            not self.legionella_cooldown_enabled
            or self.model_kind != "rc"
            or self._last_legionella_attainment_time is None
        ):
            return mask
        state = np.asarray(measurement, dtype=float).reshape(-1).copy()
        lower_boundary = self.legionella_temperature_c - self.legionella_cooldown_band_c
        expiry = self._last_legionella_attainment_time + self.legionella_cooldown_memory
        if (
            timestamp >= expiry
            or not np.isfinite(state).all()
            or float(np.array([0.75, 0.25]) @ state) < lower_boundary
        ):
            return mask
        timestep_hours = self.model_timestep.total_seconds() / 3600
        for step in range(mask.size):
            end = timestamp + (step + 1) * self.model_timestep
            if end >= expiry:
                break
            transition, _, forcing = rc_step_matrices(
                model["rc"], 0.0, disturbances[1, step],
                disturbances[2, step], timestep_hours,
            )
            state = transition @ state + forcing
            if float(np.array([0.75, 0.25]) @ state) < lower_boundary:
                break
            mask[step] = 0.0
        return mask

    def _record_legionella_mask(
        self,
        timestamp: pd.Timestamp,
        measurement: np.ndarray,
        draw_indicator: np.ndarray,
        cooldown_mask: np.ndarray,
    ) -> None:
        """Retain the numeric masks supplied to each attempted optimization."""
        weighted = float(np.array([0.75, 0.25]) @ measurement)
        for step, (draw, cooldown) in enumerate(zip(draw_indicator, cooldown_mask)):
            self._legionella_mask_history.append({
                "solved_at": pd.Timestamp(timestamp),
                "timestamp": timestamp + (step + 1) * self.model_timestep,
                "draw_indicator": float(draw),
                "cooldown_mask": float(cooldown),
                "legionella_indicator": float(draw * cooldown),
                "last_attainment_time": self._last_legionella_attainment_time,
                "measured_weighted_temperature_c": weighted,
            })

    def _forecast_legionella_indicator(
        self,
        timestamp: pd.Timestamp,
        forecast: Forecast,
        steps: int,
    ) -> np.ndarray | None:
        """Enable the bacteria penalty outside forecasted substantial draws.

        The paper disables it for a draw above 0.75 L/min in the recent
        25 minutes, or for more than 18 L drawn in the recent two hours.
        The forecast has one-minute flow values in L/min by default.
        """
        index = self._planning_index(timestamp, forecast)[:steps]
        draws = forecast.data.get("Water Heating (L/min)")
        if draws is None or len(index) != steps or len(draws.index) < 2:
            return None
        source_minutes = (
            draws.index[1] - draws.index[0]
        ).total_seconds() / 60
        if source_minutes <= 0:
            return None

        indicator = []
        for end in index + self.model_timestep:
            recent_peak = draws.loc[
                (draws.index >= end - pd.Timedelta(minutes=25))
                & (draws.index < end)
            ]
            recent_volume = draws.loc[
                (draws.index >= end - pd.Timedelta(hours=2))
                & (draws.index < end)
            ]
            if (
                recent_peak.empty
                or recent_volume.empty
                or recent_volume.isna().any()
            ):
                return None
            substantial_use = (
                (recent_peak > self.water_use_peak_l_per_min).any()
                or recent_volume.sum() * source_minutes
                > self.water_use_two_hour_volume_l
            )
            indicator.append(0.0 if substantial_use else 1.0)
        return np.asarray(indicator)

    def _planning_index(
        self, timestamp: pd.Timestamp, forecast: Forecast
    ) -> pd.DatetimeIndex:
        available = forecast.window(timestamp, self.horizon)
        desired = pd.date_range(
            timestamp,
            timestamp + self.horizon - self.model_timestep,
            freq=self.model_timestep,
        )
        return desired[desired.isin(available.index)]

    def _measurement(
        self, previous_status: Mapping[str, Any], output_names: tuple[str, ...]
    ) -> np.ndarray | None:
        values = [previous_status.get(name) for name in output_names]
        if all(value is not None for value in values):
            return np.asarray(values, dtype=float)
        if self.model_kind == "rc":
            return None
        average = previous_status.get("Hot Water Average Temperature (C)")
        if average is None:
            average = previous_status.get("Water Heating Average Temperature (C)")
        if average is None:
            return None
        return np.full(len(output_names), float(average))

    def _update_history(
        self,
        heat: float,
        disturbances: np.ndarray,
        measurement: np.ndarray,
        memory: int,
    ) -> None:
        sample = (heat, disturbances.copy(), measurement.copy())
        if not self._history:
            self._history = [sample] * memory
        else:
            self._history = [*self._history[-(memory - 1) :], sample]

    def _lifted_state(self, memory: int) -> np.ndarray:
        return np.concatenate(
            [
                np.concatenate((np.atleast_1d(heat), disturbance, measurement))
                for heat, disturbance, measurement in self._history[-memory:]
            ]
        )

    def _setpoint_control(self) -> dict[str, dict[str, Any]]:
        if self._last_setpoint is None:
            return {}
        return {self.equipment_name: {"Setpoint": self._last_setpoint}}
