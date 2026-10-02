"""Behavior checks for the water-heater bacteria-growth objective."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cvxpy as cp
import numpy as np
import pandas as pd

from examples.single_home_hems.single_home_hems.controllers.water_heater import (
    WaterHeaterPredictiveController,
)
from examples.single_home_hems.single_home_hems.forecast import Forecast
from examples.single_home_hems.single_home_hems.water_heater_rc import (
    WaterHeaterRCModel,
)


class WaterHeaterLegionellaTests(unittest.TestCase):
    def _cooldown_model(self) -> dict:
        rc = WaterHeaterRCModel(
            volume_l=170.0,
            height_m=1.2,
            ua_w_per_k=1.9,
            upper_fraction=0.5,
            upper_heat_fraction=0.3,
            cop=3.5,
            on_power_kw=0.46,
            off_power_kw=0.001,
            deadband_c=8.17,
        )
        return {
            "rc": rc,
            "A": np.eye(2),
            "E": np.zeros((2, 3)),
            "output_matrix": np.eye(2),
            "output_names": ("upper", "lower"),
            "disturbance_names": (
                "water_heater_draw_l_per_min",
                "water_heater_mains_temperature_c",
                "water_heater_tank_ambient_temperature_c",
            ),
            "tempered_draw_temperature_c": 40.0,
        }

    def _cooldown_disturbances(self, steps: int) -> np.ndarray:
        return np.vstack((
            np.zeros(steps), np.full(steps, 10.0), np.full(steps, 20.0)
        ))

    def test_cooldown_requires_observed_weighted_attainment(self) -> None:
        controller = WaterHeaterPredictiveController(legionella_cooldown_enabled=True)
        start = pd.Timestamp("2018-01-01")
        model = self._cooldown_model()
        disturbances = self._cooldown_disturbances(3)

        # Being inside the band is insufficient without previous attainment.
        controller._observe_legionella_temperature(start, np.full(2, 45.0))
        self.assertIsNone(controller._last_legionella_attainment_time)
        np.testing.assert_array_equal(
            controller._forecast_legionella_cooldown_mask(
                start, np.full(2, 45.0), disturbances, model
            ),
            np.ones(3),
        )

        # The upper sensor alone cannot grant the allowance.
        controller._observe_legionella_temperature(start, np.array([50.0, 45.0]))
        self.assertIsNone(controller._last_legionella_attainment_time)
        controller._observe_legionella_temperature(start, np.array([50.0, 46.0]))
        self.assertEqual(controller._last_legionella_attainment_time, start)
        np.testing.assert_array_equal(
            controller._forecast_legionella_cooldown_mask(
                start, np.full(2, 45.0), disturbances, model
            ),
            np.zeros(3),
        )

    def test_cooldown_memory_refreshes_only_at_measured_target(self) -> None:
        controller = WaterHeaterPredictiveController(legionella_cooldown_enabled=True)
        start = pd.Timestamp("2018-01-01")
        target = np.full(2, controller.legionella_temperature_c)
        controller._observe_legionella_temperature(start, target)
        controller._observe_legionella_temperature(
            start + pd.Timedelta(hours=1), np.full(2, 45.0)
        )
        self.assertEqual(controller._last_legionella_attainment_time, start)
        refreshed = start + pd.Timedelta(hours=2)
        controller._observe_legionella_temperature(refreshed, target)
        self.assertEqual(controller._last_legionella_attainment_time, refreshed)
        controller._observe_legionella_temperature(
            refreshed + pd.Timedelta(hours=6), np.full(2, 45.0)
        )
        self.assertIsNone(controller._last_legionella_attainment_time)

    def test_cooldown_memory_expires_inside_planning_horizon(self) -> None:
        controller = WaterHeaterPredictiveController(legionella_cooldown_enabled=True)
        attained = pd.Timestamp("2018-01-01")
        controller._observe_legionella_temperature(
            attained, np.full(2, controller.legionella_temperature_c)
        )
        start = attained + pd.Timedelta(hours=5, minutes=15)
        np.testing.assert_array_equal(
            controller._forecast_legionella_cooldown_mask(
                start,
                np.full(2, 45.0),
                self._cooldown_disturbances(4),
                self._cooldown_model(),
            ),
            [0.0, 0.0, 1.0, 1.0],
        )
        self.assertEqual(controller._last_legionella_attainment_time, attained)

    def test_passive_cooldown_crossing_ignores_forecast_draws_and_mains(self) -> None:
        controller = WaterHeaterPredictiveController(legionella_cooldown_enabled=True)
        start = pd.Timestamp("2018-01-01")
        model = self._cooldown_model()
        controller._observe_legionella_temperature(
            start, np.full(2, controller.legionella_temperature_c)
        )
        boundary = (
            controller.legionella_temperature_c - controller.legionella_cooldown_band_c
        )
        temperature = boundary + 0.12
        measurement = np.full(2, temperature)
        # For uniform nodes the RC tank follows the bulk exponential exactly.
        tau_hours = (
            model["rc"].heat_capacity_kwh_per_k / (model["rc"].ua_w_per_k / 1000)
        )
        crossing_hours = tau_hours * np.log((temperature - 20.0) / (boundary - 20.0))
        self.assertGreater(crossing_hours, 0.5)
        self.assertLess(crossing_hours, 0.75)

        disturbances = self._cooldown_disturbances(4)
        original_disturbances = disturbances.copy()
        mask = controller._forecast_legionella_cooldown_mask(
            start, measurement, disturbances, model
        )
        np.testing.assert_array_equal(mask, [0.0, 0.0, 1.0, 1.0])
        np.testing.assert_array_equal(disturbances, original_disturbances)
        disturbances[0] = 10.0
        disturbances[1] = -100.0
        np.testing.assert_array_equal(
            controller._forecast_legionella_cooldown_mask(
                start, measurement, disturbances, model
            ),
            mask,
        )

    def test_measured_drop_below_band_clears_allowance(self) -> None:
        controller = WaterHeaterPredictiveController(legionella_cooldown_enabled=True)
        start = pd.Timestamp("2018-01-01")
        controller._observe_legionella_temperature(
            start, np.full(2, controller.legionella_temperature_c)
        )
        boundary = (
            controller.legionella_temperature_c - controller.legionella_cooldown_band_c
        )
        controller._observe_legionella_temperature(
            start + pd.Timedelta(minutes=1), np.full(2, boundary)
        )
        self.assertEqual(controller._last_legionella_attainment_time, start)
        controller._observe_legionella_temperature(
            start + pd.Timedelta(minutes=2), np.full(2, boundary - 0.01)
        )
        self.assertIsNone(controller._last_legionella_attainment_time)
        controller._observe_legionella_temperature(
            start + pd.Timedelta(minutes=3), np.full(2, 45.0)
        )
        np.testing.assert_array_equal(
            controller._forecast_legionella_cooldown_mask(
                start + pd.Timedelta(minutes=3),
                np.full(2, 45.0),
                self._cooldown_disturbances(3),
                self._cooldown_model(),
            ),
            np.ones(3),
        )

    def test_cooldown_observes_between_optimization_decisions(self) -> None:
        controller = WaterHeaterPredictiveController(legionella_cooldown_enabled=True)
        start = pd.Timestamp("2018-01-01")
        controller._last_decision_time = start
        controller._last_setpoint = 45.0
        measurement = np.full(2, controller.legionella_temperature_c)
        with (
            patch.object(controller, "_load_model", return_value=self._cooldown_model()),
            patch.object(controller, "_measurement", return_value=measurement),
            patch.object(controller, "_forecast_disturbances") as forecast_disturbances,
        ):
            observed_at = start + pd.Timedelta(minutes=1)
            controller.controls_for_step(
                observed_at, {"Water Heating Mode": "Off"}, {}, SimpleNamespace()
            )
        self.assertEqual(controller._last_legionella_attainment_time, observed_at)
        forecast_disturbances.assert_not_called()

    def test_disabled_cooldown_and_state_space_preserve_indicator(self) -> None:
        start = pd.Timestamp("2018-01-01")
        self.assertFalse(WaterHeaterPredictiveController().legionella_cooldown_enabled)
        for controller in (
            WaterHeaterPredictiveController(legionella_cooldown_enabled=False),
            WaterHeaterPredictiveController(
                model_kind="state-space", legionella_cooldown_enabled=True
            ),
        ):
            with self.subTest(controller=controller):
                controller._observe_legionella_temperature(
                    start, np.full(2, controller.legionella_temperature_c)
                )
                self.assertIsNone(controller._last_legionella_attainment_time)
                np.testing.assert_array_equal(
                    controller._forecast_legionella_cooldown_mask(
                        start, np.full(2, 45.0), self._cooldown_disturbances(3), {}
                    ),
                    np.ones(3),
                )

    def test_runtime_multiplies_cooldown_by_existing_draw_indicator(self) -> None:
        controller = WaterHeaterPredictiveController(
            optimization_kind="convex", legionella_cooldown_enabled=True
        )
        attained = pd.Timestamp("2018-01-01")
        controller._observe_legionella_temperature(
            attained, np.full(2, controller.legionella_temperature_c)
        )
        start = attained + pd.Timedelta(hours=5, minutes=15)
        index = pd.date_range(start, periods=60, freq="min")
        forecast = Forecast(pd.DataFrame(
            {"Water Heating (L/min)": 0.0}, index=index
        ))
        problem = SimpleNamespace(status=cp.OPTIMAL, solve=lambda **_: None)
        heat = SimpleNamespace(value=np.zeros(4))
        with (
            patch.object(controller, "_load_model", return_value=self._cooldown_model()),
            patch.object(controller, "_measurement", return_value=np.full(2, 45.0)),
            patch.object(
                controller,
                "_forecast_disturbances",
                return_value=self._cooldown_disturbances(4),
            ),
            patch.object(controller, "_forecast_prices", return_value=np.full(4, 0.2)),
            patch.object(
                controller,
                "_forecast_legionella_indicator",
                return_value=np.array([1.0, 1.0, 0.0, 1.0]),
            ),
            patch.object(
                controller, "optimization_problem_convex", return_value=(problem, heat)
            ) as optimization,
            patch.object(controller, "_record_optimization_plan"),
        ):
            controller.controls_for_step(
                start, {"Water Heating Mode": "Off"}, {}, forecast
            )

        combined = optimization.call_args.kwargs["legionella_indicator"]
        np.testing.assert_array_equal(combined, [0.0, 0.0, 0.0, 1.0])
        self.assertTrue(np.isin(combined, (0.0, 1.0)).all())
        history = controller.legionella_mask_history
        np.testing.assert_array_equal(history["cooldown_mask"], [0.0, 0.0, 1.0, 1.0])
        np.testing.assert_array_equal(history["legionella_indicator"], combined)

    def test_precomputed_cooldown_preserves_convexity_without_predicted_attainment(self) -> None:
        controller = WaterHeaterPredictiveController(
            optimization_kind="convex",
            legionella_weight=10.0,
            legionella_cooldown_enabled=True,
        )
        start = pd.Timestamp("2018-01-01")
        model = self._cooldown_model()
        disturbances = self._cooldown_disturbances(4)
        measurement = np.full(2, 45.0)
        controller._observe_legionella_temperature(start, measurement)
        mask = controller._forecast_legionella_cooldown_mask(
            start, measurement, disturbances, model
        )
        problem, heat = controller.optimization_problem_convex(
            measurement,
            np.full(4, 0.2),
            disturbances,
            model,
            previous_heat_kw=0.0,
            legionella_indicator=mask,
        )
        self.assertTrue(problem.is_dcp())
        problem.solve(solver=cp.CLARABEL)
        self.assertEqual(problem.status, cp.OPTIMAL)
        self.assertGreater(float(heat.value[0]), 0.0)
        controller._record_optimization_plan(
            start, measurement, heat.value, np.full(4, 0.2), disturbances, model
        )
        plan = controller.optimization_plans[-1]
        predicted_weighted = 0.75 * plan["upper"] + 0.25 * plan["lower"]
        self.assertGreaterEqual(
            predicted_weighted.max(), controller.legionella_temperature_c - 1e-4
        )
        self.assertIsNone(controller._last_legionella_attainment_time)
        np.testing.assert_array_equal(
            controller._forecast_legionella_cooldown_mask(
                start, measurement, disturbances, model
            ),
            np.ones(4),
        )

    def test_cooldown_configuration_requires_positive_finite_parameters(self) -> None:
        for band in (0.0, -1.0, np.nan, np.inf):
            with self.subTest(band=band), self.assertRaises(ValueError):
                WaterHeaterPredictiveController(legionella_cooldown_band_c=band)
        for memory in (pd.Timedelta(0), pd.Timedelta(hours=-1), pd.NaT):
            with self.subTest(memory=memory), self.assertRaises(ValueError):
                WaterHeaterPredictiveController(legionella_cooldown_memory=memory)

    def test_previous_delivered_heat_reaches_convex_problem(self) -> None:
        controller = WaterHeaterPredictiveController(
            model_kind="state-space",
            optimization_kind="convex",
        )
        model = {"output_names": ("upper", "lower"), "m": 1}
        problem = SimpleNamespace(
            status=cp.OPTIMAL,
            solve=lambda **_: None,
        )
        heat = SimpleNamespace(value=np.array([0.0]))
        with (
            patch.object(controller, "_load_model", return_value=model),
            patch.object(controller, "_measurement", return_value=np.zeros(2)),
            patch.object(
                controller,
                "_forecast_disturbances",
                return_value=np.zeros((1, 2)),
            ),
            patch.object(controller, "_update_history"),
            patch.object(controller, "_lifted_state", return_value=np.zeros(2)),
            patch.object(controller, "_forecast_prices", return_value=np.zeros(2)),
            patch.object(
                controller,
                "_forecast_legionella_indicator",
                return_value=np.zeros(2),
            ),
            patch.object(
                controller,
                "optimization_problem_convex",
                return_value=(problem, heat),
            ) as optimization,
            patch.object(controller, "_record_optimization_plan"),
        ):
            controller.controls_for_step(
                pd.Timestamp("2018-01-01 00:01"),
                {
                    "Water Heating Mode": "Off",
                    "Water Heating Delivered (W)": 1750.0,
                },
                {},
                SimpleNamespace(),
            )

        self.assertEqual(
            optimization.call_args.kwargs["previous_heat_kw"],
            1.75,
        )

    def test_previous_delivered_heat_reaches_milp_problem(self) -> None:
        controller = WaterHeaterPredictiveController(
            model_kind="state-space",
            optimization_kind="milp",
        )
        model = {"output_names": ("upper", "lower"), "m": 1}
        problem = SimpleNamespace(status=cp.OPTIMAL, value=0.0)
        problem.solve = lambda **_: None
        heat = SimpleNamespace(value=np.array([0.0]))
        with (
            patch.object(controller, "_load_model", return_value=model),
            patch.object(controller, "_measurement", return_value=np.zeros(2)),
            patch.object(
                controller,
                "_forecast_disturbances",
                return_value=np.zeros((1, 2)),
            ),
            patch.object(controller, "_update_history"),
            patch.object(controller, "_lifted_state", return_value=np.zeros(2)),
            patch.object(controller, "_forecast_prices", return_value=np.zeros(2)),
            patch.object(
                controller,
                "_forecast_legionella_indicator",
                return_value=np.zeros(2),
            ),
            patch.object(
                controller,
                "optimization_problem_milp",
                return_value=(problem, heat, SimpleNamespace(value=np.zeros(2))),
            ) as optimization,
            patch.object(controller, "_record_optimization_plan"),
        ):
            controller.controls_for_step(
                pd.Timestamp("2018-01-01 00:01"),
                {
                    "Water Heating Mode": "Off",
                    "Water Heating Delivered (W)": 1750.0,
                },
                {},
                SimpleNamespace(),
            )

        self.assertEqual(
            optimization.call_args.kwargs["previous_heat_kw"],
            1.75,
        )

    def test_forecast_indicator_disables_penalty_for_draws(self) -> None:
        controller = WaterHeaterPredictiveController()
        index = pd.date_range("2018-01-01", periods=5 * 60, freq="min")
        draws = pd.Series(0.0, index=index)
        start = pd.Timestamp("2018-01-01 02:00")
        forecast = Forecast(pd.DataFrame({"Water Heating (L/min)": draws}))
        np.testing.assert_array_equal(
            controller._forecast_legionella_indicator(start, forecast, 3),
            [1.0, 1.0, 1.0],
        )

        draws.loc[start + pd.Timedelta(minutes=5)] = 1.0
        forecast = Forecast(pd.DataFrame({"Water Heating (L/min)": draws}))
        np.testing.assert_array_equal(
            controller._forecast_legionella_indicator(start, forecast, 3),
            [0.0, 0.0, 1.0],
        )
        disturbances = controller._forecast_disturbances(
            start,
            {},
            {},
            forecast,
            {"disturbance_names": ("water_heater_draw_l_per_min",)},
        )
        self.assertAlmostEqual(disturbances[0, 0], 1.0 / 15.0)

        draws[:] = 0.0
        draws.loc[start - pd.Timedelta(minutes=40) : start - pd.Timedelta(minutes=1)] = 0.5
        forecast = Forecast(pd.DataFrame({"Water Heating (L/min)": draws}))
        np.testing.assert_array_equal(
            controller._forecast_legionella_indicator(start, forecast, 3),
            [0.0, 0.0, 0.0],
        )

    def test_penalty_changes_optimized_heat_only_when_active(self) -> None:
        controller = WaterHeaterPredictiveController(
            model_kind="state-space",
            nominal_hp_capacity_kw=4.5,
            legionella_weight=10.0,
            persistence_weight=0.0,
        )
        model = {
            "A": np.eye(2),
            "B": np.ones((2, 1)),
            "E": np.zeros((2, 1)),
            "output_matrix": np.eye(2),
        }
        heat_values = []
        for indicator in (np.array([0.0]), np.array([1.0])):
            problem, heat = controller.optimization_problem_convex(
                np.array([45.0, 45.0]),
                np.array([0.2]),
                np.zeros((1, 1)),
                model,
                previous_heat_kw=0.0,
                legionella_indicator=indicator,
            )
            problem.solve(solver=cp.CLARABEL)
            self.assertEqual(problem.status, cp.OPTIMAL)
            controller._record_objective_components(
                pd.Timestamp("2018-01-01"), problem
            )
            heat_values.append(float(heat.value[0]))
        self.assertLess(heat_values[0], 1e-4)
        self.assertGreater(heat_values[1], 3.0)
        history = controller.objective_component_history
        self.assertEqual(len(history), 2)
        np.testing.assert_allclose(
            history["component_sum"],
            history["scalar_objective_cost"],
            atol=1e-9,
        )


if __name__ == "__main__":
    unittest.main()
