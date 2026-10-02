"""Scenario MPC behavior, causal inputs, and hybrid objective diagnostics."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cvxpy as cp
import numpy as np
import pandas as pd

from examples.single_home_hems.single_home_hems.controllers.water_heater import WaterHeaterPredictiveController
from examples.single_home_hems.single_home_hems.forecast import Forecast, PRICE_COLUMN
from test import test_water_heater_legionella as legionella_tests


class WaterHeaterScenarioOptimizerTests(unittest.TestCase):
    @staticmethod
    def model() -> dict:
        return {
            "A": np.eye(2), "B": np.ones((2, 1)), "E": -np.ones((2, 1)),
            "output_matrix": np.eye(2), "output_names": ("upper", "lower"),
            "disturbance_names": ("water_heater_draw_l_per_min",),
        }

    @staticmethod
    def controller(kind: str, **options) -> WaterHeaterPredictiveController:
        defaults = dict(
            model_kind="state-space", optimization_kind=kind,
            nominal_hp_capacity_kw=2, cop=1, minimum_temperature_c=40,
            comfort_weight=1, legionella_weight=0, persistence_weight=0,
            enforce_cycle_limit=False,
        )
        return WaterHeaterPredictiveController(**(defaults | options))

    def build(self, controller, disturbances, prices, **options):
        defaults = dict(previous_heat_kw=0)
        defaults.update(options)
        if controller.optimization_kind == "convex":
            return controller.optimization_problem_convex_scenario(
                np.full(2, 40), prices, disturbances, model=self.model(), **defaults
            )
        return controller.optimization_problem_milp_scenario(
            np.full(2, 40), prices, disturbances, self.model(),
            prior_on=0, recent_cycles=0, **defaults,
        )[:2]

    @staticmethod
    def solve(problem, kind):
        problem.solve(solver="CLARABEL" if kind == "convex" else "SCIPY")
        assert problem.status == cp.OPTIMAL

    def test_worst_path_comfort_turns_heat_on_despite_lower_expected_benefit(self):
        # The worst path benefits by 1 per kW, above the 0.75 energy price.
        # Averaging comfort would reduce that benefit to 0.5 and leave heat off.
        for kind in ("convex", "milp"):
            with self.subTest(kind=kind):
                controller = self.controller(kind)
                problem, heat = self.build(controller, np.array([[[0]], [[2]]]), [0.75])
                self.solve(problem, kind)
                np.testing.assert_allclose(heat.value, 2, atol=1e-6)
                self.assertAlmostEqual(problem.value, 0.375, places=6)
                controller._record_objective_components(pd.Timestamp("2018-01-01"), problem)
                components = controller.objective_component_history.iloc[0]
                self.assertAlmostEqual(components["electricity_objective_cost"], 0.375, places=6)
                self.assertAlmostEqual(components["comfort_objective_cost"], 0, places=6)
                self.assertAlmostEqual(components["objective_balance_error"], 0)

    def test_comfort_max_is_over_horizon_totals_while_other_terms_are_means(self):
        # Each path has a 2-degree deficit, but in different intervals. The
        # worst horizon total is 2, whereas summing interval maxima would be 4.
        paths = np.array([[[2, 0]], [[0, 2]]])
        planned_heat = np.array([[0, 2], [0, 0]])
        expected_components = {
            "electricity_objective_cost": 0.0375,
            "comfort_objective_cost": 0.25,
            "legionella_objective_cost": 0.075,
            "persistence_objective_cost": 0.045,
        }
        for kind in ("convex", "milp"):
            with self.subTest(kind=kind):
                controller = self.controller(
                    kind, legionella_temperature_c=42,
                    legionella_weight=0.3, persistence_weight=0.2,
                )
                problem, heat = self.build(
                    controller, paths, [0.15, 0.30], previous_heat_kw=0.8,
                    legionella_indicator=np.array([[1, 0], [0, 0]]),
                )
                problem = cp.Problem(
                    problem.objective, [*problem.constraints, heat == planned_heat]
                )
                self.solve(problem, kind)
                controller._record_objective_components(pd.Timestamp("2018-01-01"), problem)
                components = controller.objective_component_history.iloc[0]
                for name, expected in expected_components.items():
                    self.assertAlmostEqual(components[name], expected, places=7)
                self.assertAlmostEqual(problem.value, 0.4075, places=7)
                self.assertAlmostEqual(components["objective_balance_error"], 0)

    def test_first_action_is_shared_but_later_actions_can_respond_to_each_path(self):
        for kind in ("convex", "milp"):
            with self.subTest(kind=kind):
                controller = self.controller(kind, comfort_weight=10)
                problem, heat = self.build(controller, np.array([[[0, 0]], [[0, 2]]]), [0.4, 0.1])
                self.solve(problem, kind)
                self.assertEqual(heat.shape, (2, 2))
                np.testing.assert_allclose(heat.value[:, 0], 0, atol=1e-5)
                np.testing.assert_allclose(heat.value[:, 1], [0, 2], atol=1e-5)

    def test_dwell_constraints_and_objective_terms_apply_to_every_milp_path(self):
        controller = self.controller(
            "milp", comfort_weight=0.7, legionella_weight=0.3,
            persistence_weight=0.2, enforce_dwell_constraints=True,
            dwell_time_on=pd.Timedelta(minutes=15), dwell_time_off=pd.Timedelta(minutes=15),
        )
        paths = np.array([[[0, 0, 0]], [[1, 2, 1]]])
        prices = np.array([0.15, 0.3, 0.1])
        indicators = np.array([[1, 0, 1], [0, 1, 1]])
        problem, heat = self.build(
            controller, paths, prices, previous_heat_kw=0.8,
            prior_dwell_steps=1, legionella_indicator=indicators,
        )
        self.solve(problem, "milp")
        np.testing.assert_allclose(heat.value[:, 0], 0, atol=1e-8)
        expected = {name: [] for name in ("electricity", "comfort", "legionella", "persistence")}
        for scenario, h in enumerate(heat.value):
            temperatures = 40 + np.cumsum(h - paths[scenario, 0])
            expected["electricity"].append((prices * h).sum() * 0.25 / 3)
            expected["comfort"].append(0.7 * np.maximum(40 - temperatures, 0).sum() * 0.25 / 3)
            expected["legionella"].append(0.3 * (indicators[scenario] * np.maximum(controller.legionella_temperature_c - temperatures, 0)).sum() * 0.25 / 3)
            expected["persistence"].append(0.2 * np.abs(np.diff(np.r_[0.8, h])).sum() * 0.25 / 3)
        for name, values in expected.items():
            actual = controller._objective_component_expressions[f"{name}_objective_cost"].value
            expected_value = max(values) if name == "comfort" else np.mean(values)
            self.assertAlmostEqual(float(actual), float(expected_value), places=7)

    def test_scenario_masks_ignore_future_real_draws(self):
        controller = WaterHeaterPredictiveController(forecast_type="scenario")
        start = pd.Timestamp("2018-01-01 02:00")
        index = pd.date_range(start - pd.Timedelta(hours=2), periods=14 * 60, freq="min")
        data = pd.DataFrame({"Water Heating (L/min)": 0.0}, index=index)
        paths = np.zeros((2, 48))
        paths[1, 0] = 2
        original = controller._scenario_legionella_indicators(start, Forecast(data), paths)
        data.loc[data.index >= start, "Water Heating (L/min)"] = 100
        changed = controller._scenario_legionella_indicators(start, Forecast(data), paths)
        np.testing.assert_array_equal(original, changed)
        self.assertEqual(original[0, 0], 1)
        self.assertEqual(original[1, 0], 0)

    def test_control_uses_quarter_hours_and_complete_paths_without_future_draw_leakage(self):
        model = legionella_tests.WaterHeaterLegionellaTests()._cooldown_model()
        start = pd.Timestamp("2018-01-01 00:15")
        index = pd.date_range(start - pd.Timedelta(minutes=15), periods=13 * 60, freq="min")
        paths = np.zeros((2, 48))
        paths[1, 3] = 2
        provider = SimpleNamespace(scenarios_at=lambda _: SimpleNamespace(values=paths, diagnostics={}))
        data = pd.DataFrame({"Water Heating (L/min)": 0.0, "Mains Temperature (C)": 10.0, PRICE_COLUMN: 0.15}, index=index)
        outcomes = []
        for future_draw in (0.0, 100.0):
            controller = WaterHeaterPredictiveController(forecast_type="scenario", optimization_kind="convex", solver="CLARABEL")
            controller._model = model
            data.loc[data.index >= start, "Water Heating (L/min)"] = future_draw
            forecast = Forecast(data.copy(), water_scenarios=provider)
            status = {"Water Heating Mode": "Off", "upper": 45.0, "lower": 42.0, "Temperature - Indoor (C)": 20.0}
            with patch.object(controller, "_controls_for_scenarios", wraps=controller._controls_for_scenarios) as solve:
                self.assertEqual(controller.controls_for_step(start - pd.Timedelta(minutes=14), status, {}, forecast), {})
                solve.assert_not_called()
                result = controller.controls_for_step(start, status, {}, forecast)
                solve.assert_called_once()
            self.assertEqual(len(controller.solve_history), 1)
            self.assertTrue(controller.solve_history.iloc[0]["successful"])
            self.assertEqual(len(controller.optimization_plans[0]), 49)
            self.assertIn("upper_p10", controller.optimization_plans[0])
            outcomes.append((result, controller.objective_component_history.iloc[0]["scalar_objective_cost"]))
        self.assertEqual(outcomes[0][0], outcomes[1][0])
        self.assertAlmostEqual(outcomes[0][1], outcomes[1][1], places=8)

    def test_invalid_scenario_shape_is_rejected(self):
        controller = self.controller("convex")
        for paths in (np.zeros((2, 4)), np.zeros((0, 1, 4)), np.full((2, 1, 4), np.nan)):
            with self.subTest(shape=paths.shape), self.assertRaises(ValueError):
                self.build(controller, paths, np.zeros(4))


if __name__ == "__main__":
    unittest.main()
