"""EV-only Hub control holds a mode and prices the full home peak."""

from __future__ import annotations

import datetime as dt
from dataclasses import replace
import unittest
from unittest.mock import patch

import cvxpy as cp
import numpy as np
import pandas as pd

from examples.five_home_block.five_home_block.ev_equipment import BidirectionalEV
from examples.single_home_hems.single_home_hems.control_forecasts import EVScenarioSet
from examples.single_home_hems.single_home_hems.controllers.ev import EVScenarioController
from examples.single_home_hems.single_home_hems.controllers.ev_home_hub import EVHomeHubController
from examples.single_home_hems.single_home_hems.ev_hub_forecast import EVHubIntervalForecast
from examples.single_home_hems.single_home_hems.forecast import Forecast
from test.test_ev_v2h import schedule, vehicle


def issued_hub(timestamp: pd.Timestamp) -> EVHubIntervalForecast:
    rows = pd.DataFrame({
        "eligible_load_mean_kw": np.full(32, 2.0),
        "eligible_mean_kw": np.full(32, 2.0),
        "eligible_min_kw": np.full(32, 1.0),
        "hub_mean_kw": np.full(32, 5.0),
    }, index=pd.date_range(timestamp, periods=32, freq="15min"))
    return EVHubIntervalForecast(timestamp, timestamp, rows, "test_issued_hub")


class StubHubProvider:
    def __init__(self):
        self.histories = []

    def issue(self, timestamp, completed_history, horizon, inverter_kw):
        self.histories.append(completed_history.copy())
        assert completed_history.empty or completed_history.index.max() <= timestamp
        assert horizon == pd.Timedelta(hours=8)
        assert inverter_kw == 4.0
        return issued_hub(timestamp)


class StubOutsideForecast:
    method_label = "test_outside"

    def forecast_kw(self, timestamp, steps=32):
        return pd.Series(1.5, index=pd.date_range(timestamp, periods=steps, freq="15min"))


class StubWeatherOutsideForecast(StubOutsideForecast):
    def forecast_kw(self, timestamp, steps=32):
        forecast = super().forecast_kw(timestamp, steps)
        forecast.attrs.update({
            "persisted_ambient_temperature_c": -5.0,
            "persisted_ambient_temperature_available": True,
            "persisted_ambient_relative_humidity": 0.7,
            "persisted_ambient_relative_humidity_available": True,
            "weather_inputs_ready_at": timestamp,
            "persisted_weather_window_start": timestamp - pd.Timedelta(hours=1),
            "persisted_weather_window_end_exclusive": timestamp,
        })
        return forecast


class EVHomeHubControllerTests(unittest.TestCase):
    def setUp(self):
        self.ev = vehicle(formulation="milp", solver="SCIPY", horizon_hours=8,
                          control_minutes=15)
        self.mobility = schedule()
        self.provider = StubHubProvider()
        self.controller = EVHomeHubController(self.ev, self.provider, StubOutsideForecast())
        self.timestamp = self.mobility.start
        data = pd.DataFrame(index=pd.date_range(self.timestamp, periods=8 * 60, freq="min"))
        self.forecast = Forecast.from_schedule(
            data, lambda _: 0.2, ev_schedules={self.ev.vehicle_id: self.mobility},
        )
        self.status = {
            "Home Hub Electric Power (kW)": 5.0,
            f"{self.controller.equipment_name} SOC (-)": 0.5,
            f"{self.controller.equipment_name} End SOC (-)": 0.5,
            f"{self.controller.equipment_name} SOH (-)": 1.0,
            f"{self.controller.equipment_name} Capacity (kWh)": self.ev.capacity_kwh,
            f"{self.controller.equipment_name} Electric Power (kW)": 0.0,
            f"{self.controller.equipment_name} Parked": True,
        }

    def test_rejects_a_thermal_submeter_at_the_controller_boundary(self):
        detailed_status = {
            **self.status,
            "Water Heating Electric Power (kW)": 0.8,
        }
        with self.assertRaisesRegex(ValueError, "unexpected"):
            self.controller.observe_plant_step(
                self.timestamp - pd.Timedelta(minutes=1), detailed_status,
            )
        with self.assertRaisesRegex(ValueError, "unexpected"):
            self.controller.controls_for_step(
                self.timestamp, detailed_status, {}, self.forecast,
            )

    def test_mode_is_held_and_realized_plant_caps_full_v2h(self):
        c, now = self.controller, self.timestamp
        c.observe_plant_step(now - pd.Timedelta(minutes=1), self.status)
        c.observe_plant_step(now + pd.Timedelta(minutes=1), self.status)

        def choose_discharge(timestamp, soc, forecast):
            self.assertIsNone(forecast.backed_load_kw)
            self.assertIsInstance(forecast.ev_hub_intervals, EVHubIntervalForecast)
            c.held_mode, c.fallback = -1, False
            c.next_solve = timestamp + pd.Timedelta(minutes=15)

        with patch.object(c, "_solve", side_effect=choose_discharge):
            action = c.controls_for_step(now, self.status, {}, self.forecast)
        self.assertEqual(len(self.provider.histories), 1)
        self.assertEqual(self.provider.histories[0].index.tolist(), [now])
        command = action[c.equipment_name]
        self.assertEqual(command["P Setpoint"], -self.ev.hems.max_discharge_kw)
        self.assertEqual(command["Backed Load (kW)"], 0.0)

        plant = BidirectionalEV(
            ev_config=self.ev, mobility=self.mobility,
            start_time=dt.datetime(2018, 1, 1), duration=dt.timedelta(hours=1),
            time_res=dt.timedelta(minutes=1), verbosity=7, save_results=False,
        )
        plant.actual_backed_load = lambda: 2.3
        plant.update(control_signal=command)
        self.assertAlmostEqual(plant.electric_kw, -2.3)
        plant.actual_backed_load = lambda: 0.0
        plant.update(control_signal=command)
        self.assertEqual(plant.electric_kw, 0.0)

        c.held_mode = 1
        second = c.controls_for_step(
            now + pd.Timedelta(minutes=1), self.status, {}, self.forecast,
        )[c.equipment_name]
        self.assertEqual(second["P Setpoint"], self.ev.charger_power_kw)
        self.assertEqual(len(self.provider.histories), 1)

    def test_peak_uses_full_hub_plus_outside_on_all_seven_ev_paths(self):
        now = pd.Timestamp("2018-01-01 03:00")
        starts = pd.date_range(now, periods=32, freq="15min")
        hub = issued_hub(now)
        base = replace(self.forecast, ev_hub_intervals=hub)
        scenarios = EVScenarioSet(
            now, (self.mobility,) * 7, (None,) * 7,
            (base,) * 7, tuple(str(index) for index in range(7)),
        )
        paths = []
        fixed = []
        for index in range(7):
            ac = cp.Variable(32, name=f"test_ev_ac_{index}")
            fixed.append(ac == (-1.0 if index == 6 else 0.0))
            paths.append({"boundaries": starts.append(pd.DatetimeIndex([starts[-1] + pd.Timedelta(minutes=15)])),
                          "ac": ac})
        binary = cp.Variable(boolean=True, name="test_binary")
        base_problem = cp.Problem(cp.Minimize(cp.Constant(0.0)), [binary == 0, *fixed])
        base_plan = {"objective_terms": {"electricity": cp.Constant(0.0)},
                     "scenario_plans": paths}
        with patch.object(EVScenarioController, "build_problem", return_value=(base_problem, base_plan)):
            problem, plan = self.controller.build_problem(now, 0.5, scenarios)
        self.assertTrue(problem.is_mixed_integer())
        self.assertEqual(int(plan["peak_objective_active"].sum()), 16)
        np.testing.assert_allclose(plan["hub_forecast_kw"], 5.0)
        np.testing.assert_allclose(plan["outside_hub_forecast_kw"], 1.5)
        problem.solve(solver="SCIPY")
        self.assertEqual(problem.status, cp.OPTIMAL)
        self.assertAlmostEqual(plan["modeled_home_peak_kw"].value, 6.5, places=6)
        self.assertAlmostEqual(plan["objective_terms"]["modeled_home_peak"].value,
                               0.65, places=6)
        modeled = np.asarray(plan["modeled_home_power_scenarios_kw"].value)
        np.testing.assert_allclose(modeled[:6], 6.5, atol=1e-7)
        np.testing.assert_allclose(modeled[6], 5.5, atol=1e-7)

    def test_real_ev_scenario_builder_uses_issued_interval_caps(self):
        now = self.timestamp
        self.controller.outside_hub_forecast = StubWeatherOutsideForecast()
        hub = issued_hub(now)
        path_forecast = replace(self.forecast, ev_hub_intervals=hub)
        scenarios = EVScenarioSet(
            now, (self.mobility,) * 7, (None,) * 7,
            (path_forecast,) * 7, tuple(str(index) for index in range(7)),
        )
        recorded = {}
        self.controller.trace_callback = lambda kind, rows: recorded.setdefault(kind, rows.copy())

        problem, plan = self.controller.build_problem(now, 0.5, scenarios)

        self.assertTrue(problem.is_dcp())
        self.assertTrue(problem.is_mixed_integer())
        self.assertEqual(len(plan["scenario_plans"]), 7)
        self.assertIn("comfort", plan["objective_terms"])
        self.assertIn("modeled_home_peak", plan["objective_terms"])
        for path in plan["scenario_plans"]:
            np.testing.assert_allclose(path["horizon"].discharge_mean, 2.0)
            np.testing.assert_allclose(path["horizon"].discharge_min, 1.0)
        issued = recorded["forecast_inputs"]
        self.assertEqual(len(issued), 32)
        self.assertEqual(issued["lead_step"].tolist(), list(range(32)))
        np.testing.assert_allclose(issued["hub_mean_kw"], 5.0)
        np.testing.assert_allclose(issued["outside_hub_kw"], 1.5)
        self.assertTrue((issued["hub_inputs_ready_at"] <= issued["issued_at"]).all())
        np.testing.assert_allclose(issued["persisted_ambient_temperature_c"], -5.0)
        np.testing.assert_allclose(issued["persisted_ambient_relative_humidity"], 0.7)
        self.assertTrue((issued["weather_inputs_ready_at"] == now).all())
        mobility = recorded["ev_scenario_inputs"]
        self.assertEqual(set(mobility["scenario_id"]), {str(index) for index in range(7)})
        self.assertEqual(len(mobility.groupby("scenario_id")), 7)

    def test_threshold_peak_penalizes_only_worst_active_exceedance_with_dt_over_n(self):
        now = pd.Timestamp("2018-01-01 15:00")
        starts = pd.date_range(now, periods=32, freq="15min")
        rows = issued_hub(now).intervals.copy()
        rows.loc[starts[0], "hub_mean_kw"] = 30.0
        rows.loc[(rows.index.hour >= 16) & (rows.index.hour < 22), "hub_mean_kw"] = 20.0
        hub = EVHubIntervalForecast(now, now, rows, "test_threshold_hub")
        base = replace(self.forecast, ev_hub_intervals=hub)
        scenarios = EVScenarioSet(
            now, (self.mobility,) * 5, (None,) * 5,
            (base,) * 5, tuple(str(index) for index in range(5)),
        )
        boundaries = starts.append(pd.DatetimeIndex([starts[-1] + pd.Timedelta(minutes=15)]))
        paths, fixed = [], []
        for index in range(5):
            ac = cp.Variable(32, name=f"test_threshold_ev_ac_{index}")
            fixed.append(ac == 0.0)
            paths.append({"boundaries": boundaries, "ac": ac})
        binary = cp.Variable(boolean=True, name="test_threshold_binary")
        base_problem = cp.Problem(cp.Minimize(cp.Constant(0.0)), [binary == 0, *fixed])
        def base_builder(*_args):
            return base_problem, {"objective_terms": {"electricity": cp.Constant(0.0)},
                                  "scenario_plans": paths}

        with patch.object(EVScenarioController, "build_problem", side_effect=base_builder):
            all_controller = EVHomeHubController(
                self.ev, self.provider, StubOutsideForecast(),
                peak_weight_usd_per_kw=10, peak_limit_kw=20,
                peak_window="all", scenario_count=5,
            )
            all_problem, all_plan = all_controller.build_problem(now, 0.5, scenarios)
            evening_controller = EVHomeHubController(
                self.ev, self.provider, StubOutsideForecast(),
                peak_weight_usd_per_kw=10, peak_limit_kw=15,
                peak_window="evening", scenario_count=5,
            )
            evening_problem, evening_plan = evening_controller.build_problem(now, 0.5, scenarios)
        all_problem.solve(solver="SCIPY")
        evening_problem.solve(solver="SCIPY")
        self.assertEqual(all_problem.status, cp.OPTIMAL)
        self.assertEqual(evening_problem.status, cp.OPTIMAL)
        self.assertEqual(int(all_plan["peak_objective_active"].sum()), 32)
        self.assertEqual(int(evening_plan["peak_objective_active"].sum()), 24)
        self.assertAlmostEqual(all_plan["modeled_home_peak_exceedance_kw"].value, 11.5)
        self.assertAlmostEqual(evening_plan["modeled_home_peak_exceedance_kw"].value, 6.5)
        self.assertAlmostEqual(all_plan["objective_terms"]["modeled_home_peak"].value,
                               10 * 0.25 / 32 * 11.5)
        self.assertAlmostEqual(evening_plan["objective_terms"]["modeled_home_peak"].value,
                               10 * 0.25 / 32 * 6.5)


if __name__ == "__main__":
    unittest.main()
