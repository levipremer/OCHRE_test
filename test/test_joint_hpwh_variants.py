"""Joint rolling-start integration, history seeding, and exact solver parity."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

import cvxpy as cp
import numpy as np
import pandas as pd

from examples.five_home_block.five_home_block.config import load_configuration
from examples.single_home_hems.hems import HEMSController
from examples.single_home_hems.single_home_hems.controllers.hvac import HVACPredictiveController
from examples.single_home_hems.single_home_hems.controllers.joint_hems import JointHEMSPredictiveController
from examples.single_home_hems.single_home_hems.controllers.water_heater import (
    DEFAULT_RC_MODEL_PATH, WaterHeaterPredictiveController,
)
from examples.single_home_hems.single_home_hems.controllers.whole_home_joint import (
    WholeHomeJointPredictiveController,
)
from examples.single_home_hems.single_home_hems.forecast import Forecast, PRICE_COLUMN
from examples.single_home_hems.single_home_hems.rates import MISO_RTP_DOE
from examples.single_home_hems.single_home_hems.simulation import SingleHomeHEMSSimulation, select_home


ROOT = Path(__file__).resolve().parents[1]
ISSUE = pd.Timestamp("2025-11-30 00:00")
STEP = pd.Timedelta(minutes=15)


class JointHPWHVariantTests(unittest.TestCase):
    @staticmethod
    def thermal(*, rolling=True):
        water = WaterHeaterPredictiveController(
            model_path=DEFAULT_RC_MODEL_PATH,
            forecast_type="scenario", optimization_kind="milp", horizon=pd.Timedelta(hours=8),
            comfort_weight=100 if rolling else .1,
            legionella_weight=.001,
            persistence_weight=.1 if rolling else 1.,
            enforce_cycle_limit=False, enforce_dwell_constraints=False, terminal_weight=0,
            rolling_start_limit_24h=4 if rolling else None,
            rolling_start_slack_weight=100 if rolling else 0,
            terminal_cycle_reserve_weight=.1 if rolling else 0,
        )
        hvac = HVACPredictiveController(
            horizon=pd.Timedelta(hours=8), optimization_kind="mixed integer",
            enforce_degree_hour_budget=True,
        )
        return JointHEMSPredictiveController(water, hvac, peak_weight_usd_per_kw=0)

    @staticmethod
    def whole(thermal, *, backend="cvxpy"):
        options = SimpleNamespace(
            formulation="milp", v2h=True, enforce_terminal_soc=True, enable_degradation=False,
            health_soc_min=.2, health_soc_max=.8, health_soc_weight=1000,
            health_weight=1, comfort_weight=1000, comfort_soc_min=.4,
            electricity_weight=1, persistence_weight=0, enforce_cycle_limit=False,
            enforce_dwell_constraints=False, enforce_discharge_budget=False, control_minutes=15,
        )
        return WholeHomeJointPredictiveController(
            thermal, SimpleNamespace(options=options, horizon=thermal.horizon),
            rate_structure=MISO_RTP_DOE, peak_weight_usd_per_kw=0,
            solver_backend=backend,
        )

    @staticmethod
    def seed(coordinator, starts):
        for at in sorted(starts):
            for offset, mode in ((-1, "Off"), (0, "Heat Pump On"), (1, "Off")):
                measured = at + pd.Timedelta(minutes=offset)
                coordinator.observe_plant_step(measured+pd.Timedelta(minutes=1), {
                    "Time": measured, "Water Heating Mode": mode,
                })

    @staticmethod
    def model():
        return {
            "A": np.eye(2), "B": np.zeros((2, 1)), "E": np.zeros((2, 1)),
            "output_matrix": np.eye(2), "output_names": ("upper", "lower"),
            "disturbance_names": ("water_heater_draw_l_per_min",),
        }

    def inputs(self, thermal, issue, schedule):
        water = thermal.water_heater_controller
        water.model_kind = "state-space"
        water.nominal_hp_capacity_kw = 1
        water.cop = 1
        count, steps = schedule.shape
        problem, heat, _ = water.optimization_problem_milp_scenario(
            np.full(2, 50.), np.zeros(steps), np.zeros((count, 1, steps)), self.model(),
            prior_on=0, recent_cycles=0, previous_heat_kw=0, prediction_start=issue,
        )
        prepared = SimpleNamespace(
            heat=heat, electric_power_kw=heat, prices=np.zeros(steps),
            problem=cp.Problem(problem.objective, [*problem.constraints, heat == schedule]),
            objective_components=dict(water._objective_component_expressions),
        )
        hvac = SimpleNamespace(
            timestamp=issue, electric_power_kw=cp.Constant(np.zeros(steps)),
            prices=np.zeros(steps), problem=cp.Problem(cp.Minimize(0), []),
            objective_components={"comfort_objective_cost": cp.Constant(.37)},
        )
        return prepared, hvac

    @staticmethod
    def whole_problem(coordinator, water, hvac):
        count, steps = water.electric_power_kw.shape
        ev_count = 5
        paths = [dict(
            boundaries=pd.date_range(hvac.timestamp, periods=steps+1, freq=STEP),
            ac=cp.Constant(np.zeros(steps)), water_scenario_index=wi, ev_scenario_index=ei,
        ) for wi in range(count) for ei in range(ev_count)]
        plan = dict(scenario_plans=paths, scenario_shape=(count, ev_count),
                    objective_terms={"electricity": cp.Constant(.23)})
        return coordinator.optimization_problem_milp(
            water, hvac, cp.Problem(cp.Minimize(0), []), plan, np.zeros(steps),
        )

    def test_joint_and_whole_home_observe_nested_starts_before_inactive_tariff_return(self):
        for use_whole in (False, True):
            with self.subTest(whole_home=use_whole):
                thermal = self.thermal()
                coordinator = self.whole(thermal) if use_whole else thermal
                starts = [ISSUE-pd.Timedelta(hours=2), ISSUE-pd.Timedelta(hours=1)]
                # No whole-home power meter is needed for an inactive demand
                # objective, but nested thermal history must still be observed.
                self.seed(coordinator, starts)
                water = thermal.water_heater_controller
                self.assertEqual(water.observed_start_times, tuple(starts))
                self.assertEqual(len(water.solve_history), 0)
                # The next controller preparation sees the same completed
                # feedback. It must not duplicate its already-recorded start.
                water._observe_heat_pump(ISSUE, {
                    "Time": starts[-1]+pd.Timedelta(minutes=1), "Water Heating Mode": "Off",
                })
                self.assertEqual(water.observed_start_times, tuple(starts))
                if use_whole:
                    self.assertEqual(len(coordinator._recent_home_meter_kw), 0)

    def test_joint_rolling_variant_gets_24h_warmup_while_baseline_retains_16h(self):
        config = load_configuration(ROOT/"examples/five_home_block/config", repo_root=ROOT)
        home = select_home(config, "home_01")
        for rolling, expected in ((False, 16), (True, 24)):
            for use_whole in (False, True):
                with self.subTest(rolling=rolling, whole_home=use_whole):
                    thermal = self.thermal(rolling=rolling)
                    coordinator = self.whole(thermal) if use_whole else thermal
                    simulation = SingleHomeHEMSSimulation(
                        config, home, HEMSController(device_controllers=(coordinator,)),
                    )
                    self.assertEqual(simulation._effective_hvac_warmup_duration(), pd.Timedelta(hours=expected))
                    simulation.hvac_warmup_duration = pd.Timedelta(hours=30)
                    self.assertEqual(simulation._effective_hvac_warmup_duration(), pd.Timedelta(hours=30))

    def test_first_prepared_joint_issue_uses_native_start_history(self):
        thermal = self.thermal()
        water = thermal.water_heater_controller
        starts = [ISSUE-pd.Timedelta(minutes=30*number) for number in range(1, 6)]
        self.seed(thermal, starts)
        model = water._load_model()
        upper, lower = model["output_names"]
        index = pd.date_range(ISSUE-pd.Timedelta(hours=2), ISSUE+pd.Timedelta(hours=8), freq="min")
        forecast = Forecast(pd.DataFrame({
            "Water Heating (L/min)": 0., "Mains Temperature (C)": 10., PRICE_COLUMN: 0.,
        }, index=index), water_scenarios=SimpleNamespace(
            scenarios_at=lambda _: SimpleNamespace(values=np.zeros((7, 32)), diagnostics={}),
        ))
        status = {"Time": ISSUE-pd.Timedelta(minutes=1), "Water Heating Mode": "Off",
                  upper: 50., lower: 50., "Temperature - Indoor (C)": 20.}
        prepared = water.prepare_milp_scenario_step(ISSUE, status, {}, forecast)
        self.assertIsNotNone(prepared)
        problem = cp.Problem(prepared.problem.objective,
                             [*prepared.problem.constraints, prepared.heat == 0])
        problem.solve(solver="SCIPY")
        self.assertEqual(problem.status, cp.OPTIMAL)
        # Five measured starts remain throughout this first 8h forecast.
        # The new soft allowance is four, with one unused start preferred.
        self.assertAlmostEqual(float(prepared.objective_components["cycle_slack_objective_cost"].value), 25.)
        self.assertAlmostEqual(float(prepared.objective_components["terminal_cycle_reserve_objective_cost"].value), .2)
        self.assertEqual(water.observed_start_times, tuple(sorted(starts)))

    def test_both_actual_joint_assemblies_include_mean_hpwh_costs_exactly_once(self):
        for use_whole in (False, True):
            with self.subTest(whole_home=use_whole):
                thermal = self.thermal()
                self.seed(thermal, [ISSUE-pd.Timedelta(hours=n) for n in (3, 2, 1)])
                schedule = np.zeros((7, 32))
                schedule[1, 1] = 1
                schedule[1, 3:] = 1
                water, hvac = self.inputs(thermal, ISSUE, schedule)
                if use_whole:
                    coordinator = self.whole(thermal)
                    problem = self.whole_problem(coordinator, water, hvac)
                    components = coordinator._objective_components
                else:
                    problem = thermal.optimization_problem_milp(water, hvac)
                    components = thermal._objective_component_expressions
                problem.solve(solver="SCIPY")
                self.assertEqual(problem.status, cp.OPTIMAL)
                expected = {
                    "cycle_slack_objective_cost": 100*.25/32*29/7,
                    "terminal_cycle_reserve_objective_cost": .1*2/7,
                    "persistence_objective_cost": .1*.25/32*3/7,
                }
                for name, value in expected.items():
                    self.assertIs(components["hpwh_"+name], water.objective_components[name])
                    self.assertAlmostEqual(float(components["hpwh_"+name].value), value)
                self.assertAlmostEqual(problem.value, sum(expected.values())+.37+(.23 if use_whole else 0))
                self.assertNotIn("hpwh_electricity_objective_cost", components)
                np.testing.assert_allclose(water.heat.value[:, 0], 0, atol=1e-8)
                self.assertEqual(float(water.heat.value[0, 1]), 0)
                self.assertEqual(float(water.heat.value[1, 1]), 1)
                # Scenario recourse remains free after the shared first input.
                different_first = schedule.copy()
                different_first[1, 0] = 1
                water, hvac = self.inputs(thermal, ISSUE, different_first)
                invalid = (self.whole_problem(coordinator, water, hvac) if use_whole
                           else thermal.optimization_problem_milp(water, hvac))
                invalid.solve(solver="SCIPY")
                self.assertEqual(invalid.status, cp.INFEASIBLE)

    def test_native_backend_matches_cvxpy_and_reuses_graph_after_history_expires(self):
        thermal = self.thermal()
        self.seed(thermal, [ISSUE-pd.Timedelta(hours=23, minutes=45),
                            *[ISSUE-pd.Timedelta(hours=n) for n in (3, 2, 1)]])
        coordinator = self.whole(thermal, backend="sparse_native_v1")
        schedule = np.zeros((7, 32))
        schedule[:, 0] = 1
        schedule[1, 3:] = 1
        results = []
        try:
            for number, issue in enumerate((ISSUE, ISSUE+STEP)):
                water, hvac = self.inputs(thermal, issue, schedule)
                problem = self.whole_problem(coordinator, water, hvac)
                problem.solve(solver="SCIPY")
                self.assertEqual(problem.status, cp.OPTIMAL)
                reference = problem.value
                # A real backend dispatch must compile these indexed rolling
                # windows and terminal maximum, without a CVXPY fallback.
                coordinator._solve_problem_with_retries(problem)
                self.assertEqual(problem.status, cp.OPTIMAL)
                self.assertEqual(coordinator._fast_solver_metrics["backend"], "sparse_native_v1")
                self.assertNotIn("fallback_reason", coordinator._fast_solver_metrics)
                self.assertEqual(coordinator._fast_solver_metrics["native_model_reused"], number == 1)
                self.assertTrue(coordinator._fast_solver_metrics["primal_check"]["passed"])
                self.assertAlmostEqual(problem.value, reference, places=7)
                components = coordinator._objective_components
                slack = float(components["hpwh_cycle_slack_objective_cost"].value)
                reserve = float(components["hpwh_terminal_cycle_reserve_objective_cost"].value)
                self.assertAlmostEqual(slack, 100*.25/32*(36 if number == 0 else 29)/7)
                self.assertAlmostEqual(reserve, .1*8/7)
                results.append(slack)
            self.assertAlmostEqual(results[0]-results[1], 100*.25/32)
        finally:
            if coordinator._fast_solver is not None:
                coordinator._fast_solver.close()


if __name__ == "__main__":
    unittest.main()
