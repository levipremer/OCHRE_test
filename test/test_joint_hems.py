"""Joint HEMS objective and dispatch checks with hand-computable inputs."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cvxpy as cp
import numpy as np
import pandas as pd

from examples.single_home_hems.single_home_hems.controllers.hvac import HVACPredictiveController
from examples.single_home_hems.single_home_hems.controllers.joint_hems import JointHEMSPredictiveController
from examples.single_home_hems.single_home_hems.controllers.water_heater import WaterHeaterPredictiveController


class JointHEMSTests(unittest.TestCase):
    @staticmethod
    def controller(
        peak_weight_usd_per_kw: float = 0.0, peak_window: str = "all",
    ) -> JointHEMSPredictiveController:
        horizon = pd.Timedelta(hours=8)
        water = WaterHeaterPredictiveController(
            forecast_type="scenario", optimization_kind="milp", horizon=horizon,
        )
        hvac = HVACPredictiveController(
            optimization_kind="mixed integer", enforce_degree_hour_budget=True,
            horizon=horizon,
        )
        return JointHEMSPredictiveController(
            water, hvac, peak_weight_usd_per_kw=peak_weight_usd_per_kw,
            peak_window=peak_window,
        )

    def test_seven_scenario_eight_hour_objective_prices_total_power_once(self) -> None:
        joint = self.controller()
        scenarios, steps = 7, 32
        prices = np.where(np.arange(steps) < 12, 0.11, 0.29)
        hpwh_kw = 0.2 + 0.04 * np.arange(scenarios)[:, None] + 0.005 * np.arange(steps)
        hvac_kw = np.where(np.arange(steps) % 4 == 0, 1.4, 0.35)
        hpwh_power = cp.Variable((scenarios, steps))
        hvac_power = cp.Variable(steps)

        # Two paths incur their shortfall in different intervals. Maximizing
        # each full-horizon total gives one shortfall, not two stepwise maxima.
        deficits = np.zeros((scenarios, steps))
        deficits[0, 0] = 2.0
        deficits[1, 1] = 2.0
        comfort = cp.max(cp.hstack([
            0.1 * 0.25 / steps * cp.sum(deficits[path])
            for path in range(scenarios)
        ]))
        legionella_per_path = 0.01 * np.arange(1, scenarios + 1)
        persistence_per_path = 0.02 * np.arange(1, scenarios + 1)
        water = SimpleNamespace(
            electric_power_kw=hpwh_power,
            prices=prices,
            problem=cp.Problem(cp.Minimize(0), [hpwh_power == hpwh_kw]),
            objective_components={
                "electricity_objective_cost": cp.Constant(19.0),
                "comfort_objective_cost": comfort,
                "legionella_objective_cost": cp.sum(legionella_per_path) / scenarios,
                "persistence_objective_cost": cp.sum(persistence_per_path) / scenarios,
            },
        )
        hvac = SimpleNamespace(
            electric_power_kw=hvac_power,
            prices=prices.copy(),
            problem=cp.Problem(cp.Minimize(0), [hvac_power == hvac_kw]),
            objective_components={
                "electricity_objective_cost": cp.Constant(23.0),
                "comfort_objective_cost": cp.Constant(0.06),
                "persistence_objective_cost": cp.Constant(0.02),
                "degree_hour_objective_cost": cp.Constant(0.08),
            },
        )

        problem = joint.optimization_problem_milp(water, hvac)
        problem.solve(solver="SCIPY")
        self.assertEqual(problem.status, cp.OPTIMAL)
        expected_electricity = 0.25 / steps * np.sum(
            prices * (hvac_kw + hpwh_kw.mean(axis=0))
        )
        components = joint._objective_component_expressions
        self.assertEqual(
            [name for name in components if "electricity_objective_cost" in name],
            ["electricity_objective_cost"],
        )
        self.assertAlmostEqual(components["electricity_objective_cost"].value, expected_electricity)
        self.assertAlmostEqual(components["hpwh_comfort_objective_cost"].value, 0.1 * 0.25 * 2 / steps)
        self.assertAlmostEqual(components["hpwh_legionella_objective_cost"].value, legionella_per_path.mean())
        self.assertAlmostEqual(components["hpwh_persistence_objective_cost"].value, persistence_per_path.mean())
        self.assertAlmostEqual(
            problem.value,
            expected_electricity + 0.1 * 0.25 * 2 / steps
            + legionella_per_path.mean() + persistence_per_path.mean()
            + 0.06 + 0.02 + 0.08,
        )
        np.testing.assert_allclose(
            joint._total_power_scenarios_kw.value, hpwh_kw + hvac_kw[None, :],
        )

    def test_peak_epigraph_uses_largest_scenario_time_power_with_fixed_rest(self) -> None:
        hpwh_kw = np.array([[5.0, 0.0, 3.0], [2.0, 4.0, 1.0]])
        hvac_kw = np.array([0.5, 0.5, 0.5])
        rest_kw = np.array([1.0, 3.0, 0.5])
        prices = np.array([0.1, 0.2, 0.3])
        expected_modeled_home_kw = hpwh_kw + hvac_kw[None, :] + rest_kw[None, :]
        # The maximum is scenario 2, step 2. It differs from either the
        # scenario mean's maximum or the sum of per-step scenario maxima.
        expected_peak_kw = 7.5
        self.assertEqual(float(expected_modeled_home_kw.max()), expected_peak_kw)

        def prepared_steps():
            hpwh_power = cp.Variable(hpwh_kw.shape)
            hvac_power = cp.Variable(hvac_kw.shape)
            water = SimpleNamespace(
                electric_power_kw=hpwh_power,
                prices=prices,
                problem=cp.Problem(cp.Minimize(0), [hpwh_power == hpwh_kw]),
                objective_components={"electricity_objective_cost": cp.Constant(100.0)},
            )
            hvac = SimpleNamespace(
                electric_power_kw=hvac_power,
                prices=prices.copy(),
                problem=cp.Problem(cp.Minimize(0), [hvac_power == hvac_kw]),
                objective_components={"electricity_objective_cost": cp.Constant(200.0)},
            )
            return water, hvac

        baseline = self.controller()
        water, hvac = prepared_steps()
        baseline_problem = baseline.optimization_problem_milp(
            water, hvac, rest_of_home_kw=rest_kw,
            rest_of_home_method="conservative_rf_2h_p90_20d",
        )
        baseline_problem.solve(solver="SCIPY")
        self.assertEqual(baseline_problem.status, cp.OPTIMAL)
        self.assertEqual(
            baseline._objective_component_expressions[
                "modeled_home_peak_objective_cost"
            ].value,
            0.0,
        )
        np.testing.assert_allclose(
            baseline._modeled_home_power_scenarios_kw.value,
            expected_modeled_home_kw,
        )

        weighted = self.controller(peak_weight_usd_per_kw=0.005)
        water, hvac = prepared_steps()
        peak_problem = weighted.optimization_problem_milp(
            water, hvac, rest_of_home_kw=rest_kw,
            rest_of_home_method="conservative_rf_2h_p90_20d",
        )
        peak_problem.solve(solver="SCIPY")
        self.assertEqual(peak_problem.status, cp.OPTIMAL)
        np.testing.assert_allclose(
            weighted._modeled_home_power_scenarios_kw.value,
            expected_modeled_home_kw,
        )
        self.assertAlmostEqual(
            weighted._objective_component_expressions[
                "modeled_home_peak_objective_cost"
            ].value,
            0.005 * expected_peak_kw,
        )
        self.assertAlmostEqual(
            peak_problem.value - baseline_problem.value,
            0.005 * expected_peak_kw,
        )
        # The forecast is exogenous and excludes EV. It changes only the
        # peak expression; the existing tariff term still prices devices once.
        self.assertAlmostEqual(
            weighted._objective_component_expressions["electricity_objective_cost"].value,
            baseline._objective_component_expressions["electricity_objective_cost"].value,
        )

    def test_positive_peak_weight_requires_rest_forecast(self) -> None:
        joint = self.controller(peak_weight_usd_per_kw=0.005)
        water = SimpleNamespace(
            electric_power_kw=cp.Constant(np.array([[1.0, 2.0]])),
            prices=np.array([0.1, 0.1]),
        )
        hvac = SimpleNamespace(
            electric_power_kw=cp.Constant([0.0, 0.0]),
            prices=np.array([0.1, 0.1]),
        )
        with self.assertRaisesRegex(ValueError, "rest-of-home forecast is required"):
            joint.optimization_problem_milp(water, hvac)

    def test_daily_peak_masks_use_start_inclusive_end_exclusive_interval_starts(self) -> None:
        for window, start_hour, end_hour in (("tou", 15, 18), ("night", 3, 7)):
            with self.subTest(window=window):
                joint = self.controller(peak_window=window)
                timestamp = pd.Timestamp("2018-09-15") + pd.Timedelta(hours=start_hour) - pd.Timedelta(minutes=15)
                steps = (end_hour - start_hour) * 4 + 2
                mask = joint._peak_objective_mask(timestamp, steps)
                np.testing.assert_array_equal(mask, [False] + [True] * (steps - 2) + [False])
                self.assertEqual(joint.peak_window_hours, (start_hour, end_hour))
        # A horizon crossing midnight still uses the next day's local clock.
        night = self.controller(peak_window="night")
        mask = night._peak_objective_mask(pd.Timestamp("2018-09-15 23:00"), 32)
        np.testing.assert_array_equal(mask, [False] * 16 + [True] * 16)
        np.testing.assert_array_equal(
            self.controller()._peak_objective_mask(None, 32), np.ones(32, dtype=bool)
        )

    def test_windowed_peak_retains_scenario_max_and_full_horizon_electricity_normalization(self) -> None:
        for window, timestamp in (("tou", "2018-09-15 14:45"), ("night", "2018-09-15 02:45")):
            with self.subTest(window=window):
                joint = self.controller(peak_weight_usd_per_kw=0.005, peak_window=window)
                hpwh_kw = np.array([[50.0, 1.0, 3.0], [2.0, 4.0, 1.0]])
                hvac_kw = np.array([0.5, 0.5, 0.5])
                rest_kw = np.array([1.0, 3.0, 0.5])
                prices = np.array([0.1, 0.2, 0.3])
                hpwh_power, hvac_power = cp.Variable(hpwh_kw.shape), cp.Variable(hvac_kw.shape)
                water = SimpleNamespace(
                    electric_power_kw=hpwh_power, prices=prices,
                    problem=cp.Problem(cp.Minimize(0), [hpwh_power == hpwh_kw]),
                    objective_components={"comfort_objective_cost": cp.Constant(0.07)},
                )
                hvac = SimpleNamespace(
                    timestamp=pd.Timestamp(timestamp), electric_power_kw=hvac_power, prices=prices,
                    problem=cp.Problem(cp.Minimize(0), [hvac_power == hvac_kw]),
                    objective_components={"persistence_objective_cost": cp.Constant(0.08)},
                )
                problem = joint.optimization_problem_milp(water, hvac, rest_of_home_kw=rest_kw)
                problem.solve(solver="SCIPY")
                self.assertEqual(problem.status, cp.OPTIMAL)
                expected_electricity = 0.25 / 3 * np.sum(prices * (hvac_kw + hpwh_kw.mean(axis=0)))
                components = joint._objective_component_expressions
                self.assertAlmostEqual(components["electricity_objective_cost"].value, expected_electricity)
                self.assertAlmostEqual(components["modeled_home_peak_objective_cost"].value, 0.005 * 7.5)
                self.assertAlmostEqual(problem.value, expected_electricity + 0.005 * 7.5 + 0.07 + 0.08)
                joint._record_power_plan(pd.Timestamp(timestamp), water, hvac)
                np.testing.assert_array_equal(
                    joint.optimization_plans[-1]["peak_objective_active"], [False, True, True]
                )

    def test_window_with_no_overlap_has_zero_peak_cost_and_no_peak_variable(self) -> None:
        joint = self.controller(peak_weight_usd_per_kw=0.1, peak_window="tou")
        hpwh_power, hvac_power = cp.Variable((2, 32)), cp.Variable(32)
        water = SimpleNamespace(
            electric_power_kw=hpwh_power, prices=np.ones(32) * 0.2,
            problem=cp.Problem(cp.Minimize(0), [hpwh_power == 1.0]),
            objective_components={},
        )
        hvac = SimpleNamespace(
            timestamp=pd.Timestamp("2018-09-15 00:00"),
            electric_power_kw=hvac_power, prices=np.ones(32) * 0.2,
            problem=cp.Problem(cp.Minimize(0), [hvac_power == 2.0]),
            objective_components={},
        )
        problem = joint.optimization_problem_milp(water, hvac, rest_of_home_kw=np.full(32, 4.0))
        problem.solve(solver="SCIPY")
        self.assertEqual(problem.status, cp.OPTIMAL)
        self.assertEqual(joint._objective_component_expressions["modeled_home_peak_objective_cost"].value, 0.0)
        self.assertEqual(len(problem.variables()), 2)
        self.assertAlmostEqual(problem.value, 0.25 * 0.2 * 3.0)
        joint._record_power_plan(hvac.timestamp, water, hvac)
        self.assertFalse(joint.optimization_plans[-1]["peak_objective_active"].any())

    def test_peak_window_rejects_unknown_window(self) -> None:
        with self.assertRaisesRegex(ValueError, "Peak window"):
            self.controller(peak_window="afternoon")

    def test_joint_problem_preserves_water_first_action_sharing_and_recourse(self) -> None:
        joint = self.controller()
        water = joint.water_heater_controller
        water.model_kind = "state-space"
        water.nominal_hp_capacity_kw = 2.0
        water.cop = 1.0
        water.minimum_temperature_c = 40.0
        water.comfort_weight = 10.0
        water.legionella_weight = 0.0
        water.persistence_weight = 0.0
        water.enforce_cycle_limit = False
        model = {
            "A": np.eye(2), "B": np.ones((2, 1)), "E": -np.ones((2, 1)),
            "output_matrix": np.eye(2), "output_names": ("upper", "lower"),
            "disturbance_names": ("water_heater_draw_l_per_min",),
        }
        draws = np.zeros((7, 1, 2))
        draws[3:, 0, 1] = 2.0
        prices = np.array([0.4, 0.1])
        water_problem, heat, _ = water.optimization_problem_milp_scenario(
            np.full(2, 40.0), prices, draws, model,
            prior_on=0, recent_cycles=0, previous_heat_kw=0.0,
        )
        water_prepared = SimpleNamespace(
            electric_power_kw=heat,
            prices=prices,
            problem=water_problem,
            objective_components=dict(water._objective_component_expressions),
        )
        hvac_power = cp.Variable(2)
        hvac_prepared = SimpleNamespace(
            electric_power_kw=hvac_power,
            prices=prices,
            problem=cp.Problem(cp.Minimize(0), [hvac_power == [0.2, 0.4]]),
            objective_components={"electricity_objective_cost": cp.Constant(0)},
        )
        problem = joint.optimization_problem_milp(water_prepared, hvac_prepared)
        problem.solve(solver="SCIPY")
        self.assertEqual(problem.status, cp.OPTIMAL)
        np.testing.assert_allclose(heat.value[:, 0], heat.value[0, 0], atol=1e-7)
        np.testing.assert_allclose(heat.value[:3, 1], 0, atol=1e-7)
        np.testing.assert_allclose(heat.value[3:, 1], 2, atol=1e-7)

    def test_dispatch_solves_once_then_applies_both_actions(self) -> None:
        joint = self.controller()
        timestamp = pd.Timestamp("2018-01-01 00:15")
        rest_values = pd.Series(np.arange(32, dtype=float) / 10)
        rest_provider = SimpleNamespace(
            method_label="conservative_rf_2h_p90_20d",
            forecast_kw=Mock(return_value=rest_values),
        )
        water_prepared = SimpleNamespace(
            heat=cp.Constant(np.ones((7, 32))), scenario_count=7,
        )
        hvac_prepared = SimpleNamespace(
            heat_hvac=cp.Constant(np.zeros(32)),
            heat_hp=cp.Constant(np.zeros(32)),
            heat_er=cp.Constant(np.zeros(32)),
        )
        problem = SimpleNamespace(
            status=cp.OPTIMAL,
            solve=Mock(),
            variables=lambda: [SimpleNamespace(value=np.array([1.0]))],
        )
        with (
            patch.object(joint.water_heater_controller, "prepare_milp_scenario_step", return_value=water_prepared) as prepare_water,
            patch.object(joint.hvac_controller, "prepare_mixed_int_step", return_value=hvac_prepared) as prepare_hvac,
            patch.object(joint, "optimization_problem_milp", return_value=problem) as build_problem,
            patch.object(joint, "_record_solve"),
            patch.object(joint, "_record_objective"),
            patch.object(joint, "_record_power_plan"),
            patch.object(joint.water_heater_controller, "apply_milp_scenario_solution", return_value={"Water Heating": {"Setpoint": 60}}) as apply_water,
            patch.object(joint.hvac_controller, "apply_mixed_int_solution", return_value={"HVAC Heating": {"Setpoint": 22}}) as apply_hvac,
        ):
            controls = joint.controls_for_step(
                timestamp, {}, {},
                SimpleNamespace(rest_of_home_forecast=rest_provider),
            )

        problem.solve.assert_called_once_with(solver="GUROBI", TimeLimit=300.0, MIPGap=0.05, Threads=1)
        rest_provider.forecast_kw.assert_called_once_with(timestamp, steps=32)
        build_problem.assert_called_once()
        self.assertIs(build_problem.call_args.args[0], water_prepared)
        self.assertIs(build_problem.call_args.args[1], hvac_prepared)
        np.testing.assert_allclose(
            build_problem.call_args.kwargs["rest_of_home_kw"], rest_values.to_numpy()
        )
        self.assertEqual(
            build_problem.call_args.kwargs["rest_of_home_method"],
            rest_provider.method_label,
        )
        prepare_water.assert_called_once()
        prepare_hvac.assert_called_once()
        apply_water.assert_called_once_with(water_prepared)
        apply_hvac.assert_called_once_with(hvac_prepared)
        self.assertEqual(controls, {
            "Water Heating": {"Setpoint": 60},
            "HVAC Heating": {"Setpoint": 22},
        })

    def _dispatch_with_mock_problem(
        self, joint: JointHEMSPredictiveController, problem: SimpleNamespace,
    ) -> dict[str, dict]:
        water = SimpleNamespace(heat=cp.Constant(np.ones((7, 32))), scenario_count=7)
        hvac = SimpleNamespace(
            heat_hvac=cp.Constant(np.zeros(32)),
            heat_hp=cp.Constant(np.zeros(32)),
            heat_er=cp.Constant(np.zeros(32)),
        )
        with (
            patch.object(joint.water_heater_controller, "prepare_milp_scenario_step", return_value=water),
            patch.object(joint.hvac_controller, "prepare_mixed_int_step", return_value=hvac),
            patch.object(joint.hvac_controller, "_solver_metrics", return_value={
                "time_limit_hit": problem.status == cp.USER_LIMIT,
                "relative_gap": None,
            }),
            patch.object(joint, "optimization_problem_milp", return_value=problem),
            patch.object(joint, "_record_objective"),
            patch.object(joint, "_record_power_plan"),
            patch.object(joint.water_heater_controller, "apply_milp_scenario_solution", return_value={"Water Heating": {"Setpoint": 60}}),
            patch.object(joint.hvac_controller, "apply_mixed_int_solution", return_value={"HVAC Heating": {"Setpoint": 22}}),
        ):
            return joint.controls_for_step(
                pd.Timestamp("2018-01-01 00:15"), {}, {}, SimpleNamespace(),
            )

    def test_transient_solver_error_retries_same_problem_and_records_cause(self) -> None:
        joint = self.controller()
        joint.progress_callback = Mock()
        problem = SimpleNamespace(
            status=cp.OPTIMAL,
            solve=Mock(side_effect=[cp.error.SolverError("WLS renewal failed"), None]),
            variables=lambda: [SimpleNamespace(value=np.array([1.0]))],
        )
        with patch("examples.single_home_hems.single_home_hems.controllers.joint_hems.time.sleep") as sleep:
            controls = self._dispatch_with_mock_problem(joint, problem)
        self.assertEqual(problem.solve.call_count, 2)
        sleep.assert_called_once_with(60.0)
        self.assertEqual(controls["Water Heating"]["Setpoint"], 60)
        row = joint.solve_history.iloc[0]
        self.assertEqual(row["status"], cp.OPTIMAL)
        self.assertEqual(row["solve_attempts"], 2)
        self.assertEqual(row["solver_error_count"], 1)
        self.assertEqual(row["solver_error_message"], "WLS renewal failed")
        self.assertEqual(row["solver_error_messages"], [
            "SolverError: WLS renewal failed",
        ])
        self.assertEqual(joint.progress_callback.call_count, 1)

    def test_persistent_solver_error_stops_run_with_message(self) -> None:
        joint = self.controller()
        joint.progress_callback = Mock()
        problem = SimpleNamespace(
            status=None,
            solve=Mock(side_effect=cp.error.SolverError("WLS connection lost")),
            variables=lambda: [],
        )
        with patch("examples.single_home_hems.single_home_hems.controllers.joint_hems.time.sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "WLS connection lost"):
                self._dispatch_with_mock_problem(joint, problem)
        self.assertEqual(problem.solve.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [60.0, 300.0])
        row = joint.solve_history.iloc[0]
        self.assertEqual(row["status"], "solver_error")
        self.assertEqual(row["solver_error_count"], 3)
        self.assertEqual(len(row["solver_error_messages"]), 3)
        self.assertFalse(row["successful"])
        self.assertEqual(joint.progress_callback.call_count, 1)

    def test_unverified_user_limit_retries_then_fails_without_held_controls(self) -> None:
        joint = self.controller()
        joint.progress_callback = Mock()
        problem = SimpleNamespace(
            status=cp.USER_LIMIT,
            solve=Mock(),
            variables=lambda: [SimpleNamespace(value=np.array([1.0]))],
        )
        with (
            patch.object(joint.hvac_controller, "_time_limited_solution_is_feasible", return_value=False) as verify,
            patch("examples.single_home_hems.single_home_hems.controllers.joint_hems.time.sleep") as sleep,
        ):
            with self.assertRaisesRegex(RuntimeError, "without a verified feasible solution"):
                self._dispatch_with_mock_problem(joint, problem)
        self.assertEqual(problem.solve.call_count, 2)
        self.assertEqual(verify.call_count, 2)
        sleep.assert_called_once_with(5.0)
        row = joint.solve_history.iloc[0]
        self.assertEqual(row["status"], cp.USER_LIMIT)
        self.assertFalse(row["successful"])
        self.assertEqual(row["unaccepted_user_limit_retries"], 1)

    def test_unverified_user_limit_can_recover_on_retry(self) -> None:
        joint = self.controller()
        problem = SimpleNamespace(
            status=cp.USER_LIMIT,
            variables=lambda: [SimpleNamespace(value=np.array([1.0]))],
        )
        def solve(**_kwargs: object) -> None:
            problem.status = cp.OPTIMAL if problem.solve.call_count == 2 else cp.USER_LIMIT
        problem.solve = Mock(side_effect=solve)
        with (
            patch.object(joint.hvac_controller, "_time_limited_solution_is_feasible", return_value=False) as verify,
            patch("examples.single_home_hems.single_home_hems.controllers.joint_hems.time.sleep") as sleep,
        ):
            controls = self._dispatch_with_mock_problem(joint, problem)
        self.assertEqual(controls["HVAC Heating"]["Setpoint"], 22)
        self.assertEqual(problem.solve.call_count, 2)
        verify.assert_called_once()
        sleep.assert_called_once_with(5.0)
        row = joint.solve_history.iloc[0]
        self.assertEqual(row["status"], cp.OPTIMAL)
        self.assertTrue(row["successful"])
        self.assertEqual(row["unaccepted_user_limit_retries"], 1)

    def test_verified_user_limit_incumbent_is_accepted_without_retry(self) -> None:
        joint = self.controller()
        problem = SimpleNamespace(
            status=cp.USER_LIMIT,
            solve=Mock(),
            variables=lambda: [SimpleNamespace(value=np.array([1.0]))],
        )
        with (
            patch.object(joint.hvac_controller, "_time_limited_solution_is_feasible", return_value=True) as verify,
            patch("examples.single_home_hems.single_home_hems.controllers.joint_hems.time.sleep") as sleep,
        ):
            controls = self._dispatch_with_mock_problem(joint, problem)
        self.assertEqual(controls["Water Heating"]["Setpoint"], 60)
        problem.solve.assert_called_once()
        verify.assert_called_once()
        sleep.assert_not_called()
        row = joint.solve_history.iloc[0]
        self.assertEqual(row["status"], cp.USER_LIMIT)
        self.assertTrue(row["successful"])
        self.assertEqual(row["unaccepted_user_limit_retries"], 0)


if __name__ == "__main__":
    unittest.main()
