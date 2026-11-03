"""One receding-horizon MILP for the HPWH and HVAC heating equipment.

The device controllers prepare their established constraints and translate the
first optimized actions to OCHRE commands. This module owns the *one* shared
electricity term and the *one* solver call. Keeping those three responsibilities
separate makes the joint objective easy to inspect without duplicating either
device's dynamics or comfort rules.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import cvxpy as cp
import numpy as np
import pandas as pd

from ..forecast import Forecast
from .hvac import HVACPredictiveController, PreparedHVACMixedIntegerStep
from .water_heater import (
    PreparedWaterHeaterScenarioStep,
    WaterHeaterPredictiveController,
)


PEAK_WINDOW_HOURS = {"all": (0, 24), "tou": (15, 18), "night": (3, 7)}
SOLVER_ERROR_BACKOFF_SECONDS = (60.0, 300.0)
UNACCEPTED_USER_LIMIT_RETRIES = 1
UNACCEPTED_USER_LIMIT_BACKOFF_SECONDS = 5.0


@dataclass
class JointHEMSPredictiveController:
    """Coordinate scenario HPWH MILP and degree-hour HVAC MILP in one solve."""

    water_heater_controller: WaterHeaterPredictiveController
    hvac_controller: HVACPredictiveController
    solver: str = "GUROBI"
    solver_options: Mapping[str, Any] = field(
        default_factory=lambda: {"TimeLimit": 300.0, "MIPGap": 0.05, "Threads": 1}
    )
    peak_weight_usd_per_kw: float = 0.0
    peak_window: str = "all"
    progress_callback: Callable[[Mapping[str, Any]], None] | None = field(
        default=None, repr=False
    )

    _objective_component_expressions: dict[str, cp.Expression] = field(
        default_factory=dict, init=False, repr=False
    )
    _total_power_scenarios_kw: cp.Expression | None = field(
        default=None, init=False, repr=False
    )
    _modeled_home_power_scenarios_kw: cp.Expression | None = field(
        default=None, init=False, repr=False
    )
    _rest_of_home_forecast_kw: np.ndarray | None = field(
        default=None, init=False, repr=False
    )
    _rest_of_home_method_label: str | None = field(
        default=None, init=False, repr=False
    )
    _rest_input_diagnostics: dict[str, Any] | None = field(
        default=None, init=False, repr=False
    )
    _peak_objective_active: np.ndarray | None = field(
        default=None, init=False, repr=False
    )
    _solve_history: list[dict[str, Any]] = field(
        default_factory=list, init=False, repr=False
    )
    _objective_component_history: list[dict[str, Any]] = field(
        default_factory=list, init=False, repr=False
    )
    _optimization_plans: list[pd.DataFrame] = field(
        default_factory=list, init=False, repr=False
    )
    solve_durations_s: dict[str, list[float]] = field(
        default_factory=lambda: {"joint_milp": []}, init=False, repr=False
    )

    def __post_init__(self) -> None:
        water = self.water_heater_controller
        hvac = self.hvac_controller
        if water.optimization_kind != "milp" or water.forecast_type != "scenario":
            raise ValueError("Joint HEMS requires scenario MILP water-heater control")
        if hvac.optimization_kind != "mixed integer" or not hvac.enforce_degree_hour_budget:
            raise ValueError("Joint HEMS requires degree-hour mixed-integer HVAC control")
        if water.model_timestep != hvac.model_timestep or water.horizon != hvac.horizon:
            raise ValueError("Joint HEMS devices need the same horizon and model timestep")
        if self.solver != "GUROBI":
            raise ValueError("Joint HEMS currently uses the Gurobi MILP solver")
        if (
            isinstance(self.peak_weight_usd_per_kw, bool)
            or not np.isfinite(self.peak_weight_usd_per_kw)
            or self.peak_weight_usd_per_kw < 0
        ):
            raise ValueError("Peak weight must be finite and nonnegative USD/kW")
        if self.peak_window not in PEAK_WINDOW_HOURS:
            raise ValueError("Peak window must be all, tou, or night")

    @property
    def horizon(self) -> pd.Timedelta:
        return self.hvac_controller.horizon

    @property
    def peak_window_hours(self) -> tuple[int, int]:
        return PEAK_WINDOW_HOURS[self.peak_window]

    def _peak_objective_mask(
        self, timestamp: pd.Timestamp | None, steps: int,
    ) -> np.ndarray:
        """Select interval starts in the approved daily half-open window."""
        if self.peak_window == "all":
            return np.ones(steps, dtype=bool)
        if timestamp is None:
            raise ValueError("A prepared timestamp is required for a restricted peak window")
        index = pd.date_range(timestamp, periods=steps, freq=self.hvac_controller.model_timestep)
        start_hour, end_hour = self.peak_window_hours
        return np.asarray((index.hour >= start_hour) & (index.hour < end_hour), dtype=bool)

    @property
    def solve_history(self) -> pd.DataFrame:
        return pd.DataFrame(self._solve_history)

    @property
    def objective_component_history(self) -> pd.DataFrame:
        return pd.DataFrame(self._objective_component_history)

    @property
    def optimization_plans(self) -> tuple[pd.DataFrame, ...]:
        return tuple(self._optimization_plans)

    def observe_plant_step(
        self, timestamp: pd.Timestamp, status: Mapping[str, Any] | None,
    ) -> None:
        """Retain completed HPWH starts during native warm-up and operation."""
        if status is not None:
            self.water_heater_controller.observe_start_history(timestamp, status)

    def optimization_problem_milp(
        self,
        water: PreparedWaterHeaterScenarioStep,
        hvac: PreparedHVACMixedIntegerStep,
        rest_of_home_kw: np.ndarray | None = None,
        rest_of_home_method: str | None = None,
        rest_input_diagnostics: Mapping[str, Any] | None = None,
    ) -> cp.Problem:
        """Build the joint MILP with one electricity price on device power.

        Each HPWH path retains its own later decisions. HVAC has one trajectory
        across all paths. Existing non-electric objective terms and all device
        constraints come directly from the two prepared formulations.
        """
        scenario_power = water.electric_power_kw
        hvac_power = hvac.electric_power_kw
        scenarios, steps = scenario_power.shape
        if scenarios < 1 or steps < 1 or hvac_power.shape != (steps,):
            raise ValueError("Joint HEMS power expressions have incompatible shapes")
        water_prices = np.asarray(water.prices, dtype=float).reshape(-1)
        hvac_prices = np.asarray(hvac.prices, dtype=float).reshape(-1)
        if (
            water_prices.shape != (steps,)
            or hvac_prices.shape != (steps,)
            or not np.isfinite(water_prices).all()
            or not np.allclose(water_prices, hvac_prices, rtol=0, atol=1e-12)
        ):
            raise ValueError("Joint HEMS requires one aligned electricity price forecast")
        timestep_hours = self.hvac_controller.model_timestep.total_seconds() / 3600

        # The mean is over equally probable water-use paths. The same HVAC
        # power and the same price appear in every path; there is no second
        # price-weighted objective hidden in either device's non-electric terms.
        self._total_power_scenarios_kw = scenario_power + cp.vstack(
            [hvac_power for _ in range(scenarios)]
        )
        if rest_of_home_kw is None:
            if self.peak_weight_usd_per_kw > 0:
                raise ValueError("A rest-of-home forecast is required for the peak objective")
            self._rest_of_home_forecast_kw = None
            self._rest_of_home_method_label = None
            self._rest_input_diagnostics = None
            self._modeled_home_power_scenarios_kw = None
        else:
            rest = np.asarray(rest_of_home_kw, dtype=float)
            if rest.shape != (steps,) or not np.isfinite(rest).all() or np.any(rest < 0):
                raise ValueError("Rest-of-home forecast must be finite nonnegative kW at each step")
            self._rest_of_home_forecast_kw = rest.copy()
            self._rest_of_home_method_label = rest_of_home_method
            self._rest_input_diagnostics = dict(rest_input_diagnostics or {})
            self._modeled_home_power_scenarios_kw = (
                self._total_power_scenarios_kw + np.broadcast_to(rest, (scenarios, steps))
            )
        electricity_cost = (
            timestep_hours / (steps * scenarios)
            * cp.sum(cp.multiply(
                np.broadcast_to(water_prices, (scenarios, steps)),
                self._total_power_scenarios_kw,
            ))
        )
        components: dict[str, cp.Expression] = {
            "electricity_objective_cost": electricity_cost,
        }
        constraints = [*water.problem.constraints, *hvac.problem.constraints]
        self._peak_objective_active = self._peak_objective_mask(
            None if self.peak_window == "all" else hvac.timestamp, steps,
        )
        if self.peak_weight_usd_per_kw > 0 and self._peak_objective_active.any():
            peak_kw = cp.Variable(nonneg=True, name="modeled_home_peak_kw_excluding_ev")
            # Preserve the original all-horizon expression exactly. Restricted
            # windows retain the same maximum over all water-use scenarios.
            peak_power = self._modeled_home_power_scenarios_kw
            if self.peak_window != "all":
                peak_power = peak_power[:, self._peak_objective_active]
            constraints.append(peak_kw >= peak_power)
            components["modeled_home_peak_objective_cost"] = (
                self.peak_weight_usd_per_kw * peak_kw
            )
        else:
            components["modeled_home_peak_objective_cost"] = cp.Constant(0.0)
        for prefix, prepared in (("hpwh", water), ("hvac", hvac)):
            for name, expression in prepared.objective_components.items():
                if name != "electricity_objective_cost":
                    components[f"{prefix}_{name}"] = expression
        self._objective_component_expressions = components
        return cp.Problem(
            cp.Minimize(sum(components.values())),
            constraints,
        )

    def _held_controls(
        self, previous_status: Mapping[str, Any] | None
    ) -> dict[str, dict[str, Any]]:
        return {
            **self.water_heater_controller.held_controls_for_step(),
            **self.hvac_controller.held_controls_for_step(previous_status),
        }

    def _record_solve(
        self, timestamp: pd.Timestamp, duration_s: float, problem: cp.Problem,
        status: str, successful: bool, scenarios: int, *,
        solve_attempts: int = 1, solver_error_count: int = 0,
        solver_error_message: str | None = None,
        solver_error_messages: list[str] | None = None,
        unaccepted_user_limit_retries: int = 0,
    ) -> None:
        metrics = self.hvac_controller._solver_metrics(problem)
        self.solve_durations_s["joint_milp"].append(duration_s)
        self._solve_history.append({
            "timestamp": pd.Timestamp(timestamp),
            "optimization_kind": "joint_milp",
            "scenario_count": scenarios,
            "duration_s": duration_s,
            "status": status,
            "successful": successful,
            "solve_attempts": solve_attempts,
            "solver_error_count": solver_error_count,
            "solver_error_message": solver_error_message,
            "solver_error_messages": list(solver_error_messages or []),
            "unaccepted_user_limit_retries": unaccepted_user_limit_retries,
            **metrics,
        })
        if self.progress_callback is not None:
            durations = self.solve_durations_s["joint_milp"]
            relative_gaps = [
                row["relative_gap"] for row in self._solve_history
                if row.get("relative_gap") is not None
            ]
            self.progress_callback({
                **self._solve_history[-1],
                "solve_count": len(self._solve_history),
                "successful_solves": sum(row["successful"] for row in self._solve_history),
                "solve_seconds_total": sum(durations),
                "solve_seconds_max": max(durations),
                "time_limit_hits": sum(row["time_limit_hit"] for row in self._solve_history),
                "max_relative_mip_gap": max(relative_gaps, default=None),
            })

    def _record_objective(self, timestamp: pd.Timestamp, problem: cp.Problem) -> None:
        values = {
            name: float(np.asarray(expression.value).squeeze())
            for name, expression in self._objective_component_expressions.items()
        }
        if not all(np.isfinite(value) for value in values.values()):
            raise RuntimeError("Joint HEMS objective has a non-finite component")
        scalar = float(problem.value)
        self._objective_component_history.append({
            "timestamp": pd.Timestamp(timestamp),
            **values,
            "scalar_objective_cost": scalar,
            "objective_balance_error": scalar - sum(values.values()),
        })

    def _record_power_plan(
        self,
        timestamp: pd.Timestamp,
        water: PreparedWaterHeaterScenarioStep,
        hvac: PreparedHVACMixedIntegerStep,
    ) -> None:
        assert self._total_power_scenarios_kw is not None
        total = np.asarray(self._total_power_scenarios_kw.value, dtype=float)
        hpwh = np.asarray(water.electric_power_kw.value, dtype=float)
        heating = np.asarray(hvac.electric_power_kw.value, dtype=float).reshape(-1)
        if total.shape != hpwh.shape or total.shape[1] != heating.size:
            raise RuntimeError("Solved joint power plan has inconsistent dimensions")
        index = pd.date_range(timestamp, periods=heating.size,
                              freq=self.hvac_controller.model_timestep)
        plan = pd.DataFrame(index=index)
        plan.index.name = "timestamp"
        # Temporal eligibility is recorded even at weight zero for comparison.
        plan["peak_objective_active"] = self._peak_objective_mask(timestamp, heating.size)
        plan["electricity_price_usd_per_kwh"] = np.asarray(water.prices, dtype=float)
        plan["hvac_electric_power_kw"] = heating
        plan["hpwh_mean_electric_power_kw"] = hpwh.mean(axis=0)
        plan["total_mean_electric_power_kw"] = total.mean(axis=0)
        if self._rest_of_home_forecast_kw is not None:
            rest = self._rest_of_home_forecast_kw
            plan["rest_of_home_forecast_kw"] = rest
            plan["modeled_home_mean_electric_power_kw_excluding_ev"] = total.mean(axis=0) + rest
            plan["rest_of_home_forecast_method"] = self._rest_of_home_method_label
            for name, value in (self._rest_input_diagnostics or {}).items():
                plan[name] = value
        for scenario in range(total.shape[0]):
            plan[f"hpwh_scenario_{scenario + 1:02d}_electric_power_kw"] = hpwh[scenario]
            plan[f"total_scenario_{scenario + 1:02d}_electric_power_kw"] = total[scenario]
            if self._rest_of_home_forecast_kw is not None:
                plan[f"modeled_home_scenario_{scenario + 1:02d}_power_kw_excluding_ev"] = (
                    total[scenario] + self._rest_of_home_forecast_kw
                )
        plan.attrs["solved_at"] = pd.Timestamp(timestamp)
        self._optimization_plans.append(plan)

    def controls_for_step(
        self,
        timestamp: pd.Timestamp,
        previous_status: Mapping[str, Any] | None,
        current_schedule: Mapping[str, Any],
        forecast: Forecast,
    ) -> dict[str, dict[str, Any]]:
        """Update both memories and issue both commands from one MILP solve."""
        water = self.water_heater_controller.prepare_milp_scenario_step(
            timestamp, previous_status, current_schedule, forecast
        )
        hvac = self.hvac_controller.prepare_mixed_int_step(
            timestamp, previous_status, current_schedule, forecast
        )
        if water is None or hvac is None:
            return self._held_controls(previous_status)
        rest_provider = getattr(forecast, "rest_of_home_forecast", None)
        rest_kw = None
        rest_method = None
        rest_input_diagnostics = None
        if rest_provider is not None:
            steps = int(self.horizon / self.hvac_controller.model_timestep)
            rest_series = rest_provider.forecast_kw(timestamp, steps=steps)
            rest_kw = rest_series.to_numpy(dtype=float)
            rest_method = rest_provider.method_label
            rest_input_diagnostics = rest_series.attrs
        problem = self.optimization_problem_milp(
            water, hvac, rest_of_home_kw=rest_kw, rest_of_home_method=rest_method,
            rest_input_diagnostics=rest_input_diagnostics,
        )
        started = time.perf_counter()
        last_solver_error: cp.error.SolverError | None = None
        solver_error_messages: list[str] = []
        solver_error_count = 0
        unaccepted_user_limit_retries = 0
        solve_attempts = 0
        while True:
            solve_attempts += 1
            try:
                problem.solve(solver=self.solver, **dict(self.solver_options))
            except cp.error.SolverError as error:
                last_solver_error = error
                solver_error_count += 1
                detail = f"{type(error).__name__}: {error}"
                cause = error.__cause__ or error.__context__
                if cause is not None:
                    detail += f"; caused by {type(cause).__name__}: {cause}"
                solver_error_messages.append(detail)
                if solver_error_count > len(SOLVER_ERROR_BACKOFF_SECONDS):
                    status = "solver_error"
                    successful = False
                    print(
                        f"Joint HEMS solver at {timestamp}: attempt "
                        f"{solve_attempts} failed: {detail}; retries exhausted",
                        flush=True,
                    )
                    break
                delay = SOLVER_ERROR_BACKOFF_SECONDS[solver_error_count - 1]
                print(
                    f"Joint HEMS solver at {timestamp}: attempt "
                    f"{solve_attempts} failed: {detail}; retrying in {delay:g} s",
                    flush=True,
                )
                time.sleep(delay)
            else:
                status = str(problem.status)
                values_ready = (
                    water.heat.value is not None
                    and all(expression.value is not None for expression in (
                        hvac.heat_hvac, hvac.heat_hp, hvac.heat_er,
                    ))
                )
                time_limited_incumbent = (
                    status == cp.USER_LIMIT and values_ready
                    and self.hvac_controller._time_limited_solution_is_feasible(problem)
                )
                successful = (
                    status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
                    or time_limited_incumbent
                ) and values_ready
                if successful:
                    successful = all(
                        variable.value is not None and np.isfinite(variable.value).all()
                        for variable in problem.variables()
                    )
                if (
                    status == cp.USER_LIMIT and not successful
                    and unaccepted_user_limit_retries < UNACCEPTED_USER_LIMIT_RETRIES
                ):
                    unaccepted_user_limit_retries += 1
                    time.sleep(UNACCEPTED_USER_LIMIT_BACKOFF_SECONDS)
                    continue
                break
        duration = time.perf_counter() - started
        self._record_solve(
            timestamp, duration, problem, status, successful,
            water.scenario_count, solve_attempts=solve_attempts,
            solver_error_count=solver_error_count,
            solver_error_message=(
                str(last_solver_error) if last_solver_error is not None else None
            ),
            solver_error_messages=solver_error_messages,
            unaccepted_user_limit_retries=unaccepted_user_limit_retries,
        )
        if status == "solver_error":
            raise RuntimeError(
                f"Joint HEMS solver failed at {timestamp} after {solve_attempts} "
                f"attempts: {last_solver_error}"
            ) from last_solver_error
        if status == cp.USER_LIMIT and not successful:
            raise RuntimeError(
                f"Joint HEMS reached its configured solver time limit at "
                f"{timestamp} on {solve_attempts} attempts without a verified "
                "feasible solution"
            )
        if not successful:
            return self._held_controls(previous_status)
        self._record_objective(timestamp, problem)
        self._record_power_plan(timestamp, water, hvac)
        return {
            **self.water_heater_controller.apply_milp_scenario_solution(water),
            **self.hvac_controller.apply_mixed_int_solution(hvac),
        }
