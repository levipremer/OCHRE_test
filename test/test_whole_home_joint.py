"""Focused checks for the three-device, one-problem coordinator."""

from types import SimpleNamespace
from dataclasses import replace
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cvxpy as cp
import gurobipy as gp
import numpy as np
import pandas as pd

from examples.single_home_hems.single_home_hems.controllers.whole_home_joint import (
    WINTER_TOU_PEAK_WINDOW, WholeHomeJointPredictiveController,
    _average_ev_power_on_thermal_grid,
    _predicted_hvac_temperature_c,
    _predicted_water_outputs_c,
)
from examples.single_home_hems.single_home_hems.control_forecasts import EVScenarioSet
from examples.single_home_hems.single_home_hems.backed_loads import backed_power
from examples.single_home_hems.single_home_hems.controllers.ev import EVMPCController
from examples.single_home_hems.single_home_hems.controllers.ev_scenario import build_joint_ev_scenario_problem
from examples.single_home_hems import run_whole_home_joint
from examples.single_home_hems.single_home_hems.water_heater_rc import (
    WaterHeaterRCModel,
    rc_step_matrices,
)
from test.test_ev_v2h import forecast, schedule, vehicle


class WholeHomeJointTests(unittest.TestCase):
    def test_sparse_backend_accepts_only_the_three_validated_study_rates(self):
        original = self.make_controller()
        for rate in ("tou-winter", "tou-winter+demand+CPP", "rtp-miso+doe"):
            with self.subTest(rate=rate):
                selected = replace(original, rate_structure=rate, solver_backend="sparse_native_v1")
                self.assertEqual(selected.solver_backend, "sparse_native_v1")
                self.assertEqual(selected.rate_structure, rate)
        for rate in ("legacy", "tou-winter+demand"):
            with self.subTest(rate=rate), self.assertRaisesRegex(ValueError, "validated only"):
                replace(original, rate_structure=rate, solver_backend="sparse_native_v1")
        self.assertEqual(original.solver_backend, "cvxpy")

    def make_controller(self, peak_weight=10, doe_panel_limit_kw=None,
                        doe_grid_limit_kw=None):
        hvac = SimpleNamespace(model_timestep=pd.Timedelta(minutes=15))
        joint = SimpleNamespace(solver="GUROBI", hvac_controller=hvac,
                                water_heater_controller=SimpleNamespace(), horizon=pd.Timedelta(hours=8))
        options = SimpleNamespace(formulation="milp", v2h=True, enforce_terminal_soc=True,
                                  enable_degradation=False, health_soc_min=.2, health_soc_max=.8,
                                  health_soc_weight=1000, health_weight=1, comfort_weight=1000,
                                  comfort_soc_min=.4, electricity_weight=1, persistence_weight=0,
                                  enforce_cycle_limit=False, enforce_dwell_constraints=False,
                                  enforce_discharge_budget=False, control_minutes=15)
        ev = SimpleNamespace(options=options, horizon=pd.Timedelta(hours=8))
        return WholeHomeJointPredictiveController(
            joint, ev, peak_weight_usd_per_kw=peak_weight,
            doe_panel_limit_kw=doe_panel_limit_kw,
            doe_grid_limit_kw=doe_grid_limit_kw,
        )

    @staticmethod
    def winter_peak_inputs(start, ev_profiles):
        steps = 32
        water = SimpleNamespace(
            electric_power_kw=cp.Constant(np.zeros((7, steps))),
            prices=np.zeros(steps),
            problem=cp.Problem(cp.Minimize(0), [cp.Variable(boolean=True) == 1]),
            objective_components={},
        )
        hvac = SimpleNamespace(
            electric_power_kw=cp.Constant(np.zeros(steps)),
            prices=np.zeros(steps), timestamp=start,
            problem=cp.Problem(cp.Minimize(0), []), objective_components={},
        )
        boundaries = pd.date_range(start, periods=steps + 1, freq="15min")
        paths = [
            dict(boundaries=boundaries, ac=cp.Constant(ev_profiles[ev_index]),
                 water_scenario_index=water_index, ev_scenario_index=ev_index)
            for water_index in range(7) for ev_index in range(5)
        ]
        ev_plan = dict(scenario_plans=paths, scenario_shape=(7, 5), objective_terms={})
        return water, hvac, cp.Problem(cp.Minimize(0), []), ev_plan

    def test_winter_demand_uses_aligned_means_observed_half_and_existing_scaling(self):
        controller = self.make_controller(peak_weight=.71)
        controller.peak_window = WINTER_TOU_PEAK_WINDOW
        start = pd.Timestamp("2025-01-06 17:15", tz="America/Indiana/Indianapolis")
        with self.assertRaisesRegex(ValueError, "15 consecutive observed"):
            controller._observed_first_half_kw(start)
        for minute in pd.date_range(start - pd.Timedelta(minutes=15), periods=15, freq="min"):
            controller.observe_plant_step(minute, {"Total Electric Power (kW)": 10.0})
        observed_kw = controller._observed_first_half_kw(start)
        self.assertEqual(observed_kw, 10.0)
        profiles = np.zeros((5, 32))
        profiles[0, 1:3] = [10.0, 0.0]
        profiles[1, 1:3] = [6.0, 6.0]
        inputs = self.winter_peak_inputs(start, profiles)
        problem = controller.optimization_problem_milp(
            *inputs, np.zeros(32), observed_first_half_kw=observed_kw,
        )
        self.assertTrue(problem.is_mixed_integer())
        self.assertEqual(problem.constraints[-1].shape, (35, 8))
        self.assertEqual(controller._peak_objective_active.sum(), 15)
        self.assertEqual(controller._peak_demand_block_starts[0], start - pd.Timedelta(minutes=15))
        self.assertEqual(controller._peak_observed_first_half_kw, 10.0)
        problem.solve(solver="SCIPY")
        self.assertEqual(problem.status, cp.OPTIMAL)
        self.assertAlmostEqual(
            controller._objective_components["modeled_home_peak_objective_cost"].value,
            .71 * 6.0,
        )

        higher_observed_problem = controller.optimization_problem_milp(
            *inputs, np.zeros(32), observed_first_half_kw=14.0,
        )
        higher_observed_problem.solve(solver="SCIPY")
        self.assertEqual(higher_observed_problem.status, cp.OPTIMAL)
        self.assertAlmostEqual(
            controller._objective_components["modeled_home_peak_objective_cost"].value,
            .71 * 7.0,
        )

        controller.peak_limit_kw = 5.5
        threshold_problem = controller.optimization_problem_milp(
            *inputs, np.zeros(32), observed_first_half_kw=observed_kw,
        )
        threshold_problem.solve(solver="SCIPY")
        self.assertEqual(threshold_problem.status, cp.OPTIMAL)
        self.assertAlmostEqual(
            controller._objective_components["modeled_home_peak_objective_cost"].value,
            .71 * (.25 / 32) * .5,
        )

    def test_winter_demand_excludes_block_extending_beyond_forecast(self):
        controller = self.make_controller(peak_weight=.71)
        controller.peak_window = WINTER_TOU_PEAK_WINDOW
        start = pd.Timestamp("2025-01-05 23:15", tz="America/Indiana/Indianapolis")
        profiles = np.zeros((5, 32))
        profiles[0, -1] = 100.0  # Monday 07:00–07:15 lacks a forecast 07:15–07:30 half.
        inputs = self.winter_peak_inputs(start, profiles)
        problem = controller.optimization_problem_milp(*inputs, np.zeros(32))
        self.assertEqual(problem.constraints[-1].shape, (35, 2))
        self.assertFalse(controller._peak_objective_active[-1])
        self.assertEqual(
            [block.strftime("%H:%M") for block in controller._peak_demand_block_starts],
            ["06:00", "06:30"],
        )
        problem.solve(solver="SCIPY")
        self.assertEqual(problem.status, cp.OPTIMAL)
        self.assertAlmostEqual(
            controller._objective_components["modeled_home_peak_objective_cost"].value, 0.0,
        )

    def test_winter_peak_mask_combines_weekday_morning_and_evening_only(self):
        controller = self.make_controller()
        controller.peak_window = WINTER_TOU_PEAK_WINDOW
        timezone = "America/Indiana/Indianapolis"
        friday_morning = controller._peak_mask(
            pd.Timestamp("2025-01-03 05:00", tz=timezone), 32,
        )
        friday_evening = controller._peak_mask(
            pd.Timestamp("2025-01-03 16:00", tz=timezone), 32,
        )
        saturday_morning = controller._peak_mask(
            pd.Timestamp("2025-01-04 05:00", tz=timezone), 32,
        )
        monday_morning = controller._peak_mask(
            pd.Timestamp("2025-01-06 05:00", tz=timezone), 32,
        )
        np.testing.assert_array_equal(
            friday_morning, [False] * 4 + [True] * 8 + [False] * 20,
        )
        np.testing.assert_array_equal(
            friday_evening, [False] * 4 + [True] * 16 + [False] * 12,
        )
        self.assertFalse(saturday_morning.any())
        np.testing.assert_array_equal(monday_morning, friday_morning)

    def test_issued_joint_trace_records_exact_prepared_inputs_before_solve(self):
        controller = self.make_controller()
        start = pd.Timestamp("2025-12-01 16:00:00-05:00")
        starts = pd.date_range(start, periods=32, freq="15min")
        minute_index = pd.date_range(start, periods=480, freq="min")
        weather = pd.DataFrame({
            "Ambient Dry Bulb (C)": np.full(480, -2.0),
            "Ambient Relative Humidity (-)": np.full(480, .8),
            "GHI (W/m^2)": np.full(480, 10.0),
            "DNI (W/m^2)": np.full(480, 20.0),
            "DHI (W/m^2)": np.full(480, 5.0),
        }, index=minute_index)
        water_disturbances = np.zeros((7, 2, 32))
        water_disturbances[:, 0, :] = .5
        water_disturbances[:, 1, :] = 10.0
        water = SimpleNamespace(
            disturbances=water_disturbances,
            source_draws_l_per_min=np.ones((7, 32)),
            prices=np.full(32, .3),
            model={"disturbance_names": (
                "water_heater_draw_l_per_min", "water_heater_mains_temperature_c",
            ), "rc": SimpleNamespace(cop=2.5)},
            initial_state=np.array([45.0, 40.0]),
        )
        hvac = SimpleNamespace(
            disturbances=np.vstack((np.full(32, -2.0), np.ones(32))),
            model={"disturbance_names": ("outdoor_temperature_c", "day_indicator")},
            initial_state=np.array([20.0]),
            prices=np.full(32, .3),
            cop=np.full(32, 2.5),
            setpoints=np.full(32, 21.0),
            lower=np.full(32, 19.5),
            upper=np.full(32, 22.5),
            historical_degree_hours=.25,
        )
        controller.water_heater_controller._scenario_diagnostics_history = [
            {"timestamp": start, "candidate_ids": [f"water_{i}" for i in range(7)]}
        ]
        controller.water_heater_controller.model_kind = "rc"
        controller.ev_controller.current_capacity = 51.0
        controller._peak_objective_active = np.ones(32, dtype=bool)
        ev_horizon = SimpleNamespace(
            t=pd.date_range(start, periods=33, freq="15min"),
            dt=np.full(32, .25), z=np.ones(32, dtype=int),
            trip_kwh=np.zeros(32), group=np.arange(32),
            departure_indices=np.array([], dtype=int),
            price=np.full(32, .3), backed_cost=np.full(32, .6),
            discharge_mean=np.full(32, 2.0), discharge_min=np.full(32, 1.0),
            N=32,
        )
        prediction = EVScenarioSet(
            start, tuple([object()] * 5), tuple([None] * 5),
            tuple([object()] * 5), tuple(f"ev_{i}" for i in range(5)),
        )
        ev_plan = {
            "scenario_plans": [{"horizon": ev_horizon} for _ in range(35)],
            "scenario_ids": [f"water_{w}|ev_{e}" for w in range(7) for e in range(5)],
        }
        hub = pd.Series(np.full(32, 1.0), index=starts)
        outside = pd.Series(np.full(32, 3.0), index=starts)
        record = controller._issued_decision_trace(
            start, SimpleNamespace(data=weather, weather_archive=None),
            water, hvac, prediction, ev_plan, hub, outside, None, .4,
        )
        self.assertEqual(record["water_heater"]["raw_selected_draw_l_per_min"][0, 0], 1.0)
        self.assertEqual(
            record["water_heater"]["effective_disturbances_by_scenario"][0][
                "water_heater_draw_l_per_min"][0], .5,
        )
        self.assertEqual(record["hvac"]["disturbances_by_name"]["outdoor_temperature_c"][0], -2)
        self.assertEqual(record["hvac"]["price_usd_per_kwh"][0], .3)
        self.assertEqual(record["water_heater"]["price_usd_per_kwh"][0], .3)
        self.assertEqual(record["water_heater"]["cop"], 2.5)
        self.assertEqual(record["weather_15min_means"]["DNI (W/m^2)"][0], 20)
        self.assertEqual(record["ev_mobility_scenarios"][0]["probability"], .2)
        self.assertEqual(len(record["joint_scenario_ids"]), 35)
        self.assertAlmostEqual(record["joint_scenario_probability"], 1 / 35)
        self.assertAlmostEqual(record["ev_fixed_backed_tariff_cost_usd"], .15)
        with tempfile.TemporaryDirectory() as temporary:
            controller.decision_trace_directory = Path(temporary)
            path = controller._write_decision_trace(start, record)
            issued = json.loads(path.read_text())
            self.assertEqual(issued["status"], "issued")
            self.assertIsNone(issued["objective_components"])
            self.assertEqual(len(issued["target_starts"]), 32)
            record.update(status="infeasible_inaccurate", successful=False, fallback=True)
            controller._replace_decision_trace(path, record)
            fallback = json.loads(path.read_text())
            self.assertTrue(fallback["fallback"])
            self.assertIsNone(fallback["objective_components"])

    def test_solved_thermal_trace_reconstructs_prepared_dynamics(self):
        hvac = SimpleNamespace(
            initial_state=np.array([10.0]),
            heat_hvac=cp.Constant([4.0, 5.0]),
            disturbances=np.array([[2.0, 3.0]]),
            model={"A": np.array([[1.0]]), "B": np.array([[1.0]]),
                   "E": np.array([[1.0]]), "output_matrix": np.array([[1.0]])},
        )
        np.testing.assert_allclose(_predicted_hvac_temperature_c(hvac), [10, 16, 24])

        rc = WaterHeaterRCModel(
            volume_l=190, height_m=1.2, ua_w_per_k=3,
            upper_fraction=.5, upper_heat_fraction=.6, cop=2.5,
            on_power_kw=1.5, off_power_kw=.01, deadband_c=4,
        )
        disturbances = np.array([[[.2, .4], [12.0, 12.0], [20.0, 20.0]],
                                 [[.6, .1], [12.0, 12.0], [20.0, 20.0]]])
        water = SimpleNamespace(
            initial_state=np.array([45.0, 40.0]),
            heat=cp.Constant(np.array([[1.0, 2.0], [1.0, 3.0]])),
            disturbances=disturbances,
            model={"rc": rc, "output_matrix": np.eye(2),
                   "output_names": ("upper_temperature_c", "lower_temperature_c")},
        )
        outputs = _predicted_water_outputs_c(water, "rc", pd.Timedelta(minutes=15))
        self.assertEqual(len(outputs), 2)
        for scenario, heat in enumerate(water.heat.value):
            state = water.initial_state.copy()
            for step, heat_kw in enumerate(heat):
                transition, gain, forcing = rc_step_matrices(
                    rc, *disturbances[scenario, :, step], .25,
                )
                state = transition @ state + gain * heat_kw + forcing
                self.assertAlmostEqual(
                    outputs[scenario]["upper_temperature_c"][step + 1], state[0],
                )
                self.assertAlmostEqual(
                    outputs[scenario]["lower_temperature_c"][step + 1], state[1],
                )

    def test_ev_power_uses_time_weighted_mobility_blocks(self):
        start = pd.Timestamp("2018-01-03 03:00")
        plan = dict(boundaries=pd.DatetimeIndex([
            start, start + pd.Timedelta(minutes=5),
            start + pd.Timedelta(minutes=15), start + pd.Timedelta(minutes=30),
        ]), ac=cp.Constant([6.0, -3.0, 4.0]))
        result = _average_ev_power_on_thermal_grid(plan, start, 2, pd.Timedelta(minutes=15))
        np.testing.assert_allclose(result.value, [0.0, 4.0], atol=1e-12)

    def test_soft_panel_limit_counts_charging_but_not_v2h_as_negative_load(self):
        controller = self.make_controller(peak_weight=0, doe_panel_limit_kw=38)
        start = pd.Timestamp("2025-12-01 00:00", tz="America/Indiana/Indianapolis")
        steps = 32
        water = SimpleNamespace(
            electric_power_kw=cp.Constant(np.zeros((7, steps))),
            prices=np.zeros(steps),
            problem=cp.Problem(cp.Minimize(0), [cp.Variable(boolean=True) == 1]),
            objective_components={},
        )
        hvac = SimpleNamespace(
            electric_power_kw=cp.Constant(np.zeros(steps)),
            prices=np.zeros(steps), timestamp=start,
            problem=cp.Problem(cp.Minimize(0), []), objective_components={},
        )
        boundaries = pd.date_range(start, periods=steps + 1, freq="15min")

        def make_plan(charge_first_kw):
            paths = []
            for water_index in range(7):
                for ev_index in range(5):
                    charge = np.zeros(steps)
                    ac = np.full(steps, -9.0)
                    if water_index == 6 and ev_index == 4:
                        charge[0] = charge_first_kw
                        ac[0] = charge_first_kw
                    paths.append(dict(
                        boundaries=boundaries, ac=cp.Constant(ac),
                        charge=cp.Constant(charge),
                        water_scenario_index=water_index,
                        ev_scenario_index=ev_index,
                    ))
            return dict(scenario_plans=paths, scenario_shape=(7, 5), objective_terms={})

        background = np.full(steps, 37.0)
        feasible = controller.optimization_problem_milp(
            water, hvac, cp.Problem(cp.Minimize(0), []), make_plan(0), background,
        )
        self.assertEqual(feasible.constraints[-1].shape, (35, 32))
        feasible.solve(solver="SCIPY")
        self.assertEqual(feasible.status, cp.OPTIMAL)
        self.assertAlmostEqual(controller._modeled_home_power_scenarios_kw.value[0, 0], 28)
        self.assertAlmostEqual(controller._modeled_panel_load_scenarios_kw.value[0, 0], 37)

        charging = controller.optimization_problem_milp(
            water, hvac, cp.Problem(cp.Minimize(0), []), make_plan(2), background,
        )
        charging.solve(solver="SCIPY")
        self.assertEqual(charging.status, cp.OPTIMAL)
        self.assertAlmostEqual(controller._doe_panel_slack_kw.value[0], 1.0)
        self.assertAlmostEqual(controller._doe_panel_slack_kw.value[1:].sum(), 0.0)
        self.assertAlmostEqual(
            controller._objective_components["doe_panel_violation_objective_cost"].value,
            1000 * .25 / 32,
        )

    def test_doe_grid_slack_is_shared_across_scenarios_and_windowed(self):
        controller = self.make_controller(peak_weight=0, doe_grid_limit_kw=15)
        start = pd.Timestamp("2025-12-01 05:45", tz="America/Indiana/Indianapolis")
        profiles = np.zeros((5, 32))
        profiles[0, 1] = 2.0   # 06:00: one scenario exceeds the 15 kW envelope.
        profiles[0, 13] = 20.0  # 09:00: outside the envelope.
        inputs = self.winter_peak_inputs(start, profiles)
        background = np.full(32, 14.0)
        problem = controller.optimization_problem_milp(*inputs, background)
        self.assertEqual(int(controller._doe_grid_active.sum()), 12)
        self.assertEqual(problem.constraints[-1].shape, (35, 12))
        problem.solve(solver="SCIPY")
        self.assertEqual(problem.status, cp.OPTIMAL)
        self.assertAlmostEqual(controller._doe_grid_slack_kw.value.sum(), 1.0)
        self.assertAlmostEqual(
            controller._objective_components["doe_grid_violation_objective_cost"].value,
            10 * .25 / 32,
        )
        self.assertFalse(controller._doe_grid_mask(
            pd.Timestamp("2025-12-06 05:45", tz="America/Indiana/Indianapolis"), 32,
        ).any())

    def test_panel_limit_requires_finite_positive_value(self):
        with self.assertRaisesRegex(ValueError, "panel limit"):
            self.make_controller(doe_panel_limit_kw=0)
        with self.assertRaisesRegex(ValueError, "panel limit"):
            self.make_controller(doe_panel_limit_kw=float("nan"))

    def test_paired_peak_and_existing_objective_terms(self):
        controller = self.make_controller()
        start = pd.Timestamp("2018-01-03 02:45")
        steps = 32
        water_kw = cp.Constant(np.repeat(np.arange(1., 8.)[:, None], steps, axis=1))
        binary = cp.Variable(boolean=True)
        water = SimpleNamespace(
            electric_power_kw=water_kw, prices=np.full(steps, .15),
            problem=cp.Problem(cp.Minimize(0), [binary == 1]),
            objective_components={"electricity_objective_cost": cp.Constant(999),
                                  "comfort_objective_cost": cp.Constant(1.25)},
        )
        hvac = SimpleNamespace(
            electric_power_kw=cp.Constant(np.full(steps, 2.)),
            heat_er=cp.Constant(np.zeros(steps)),
            prices=np.full(steps, .15), timestamp=start,
            problem=cp.Problem(cp.Minimize(0), []),
            objective_components={"electricity_objective_cost": cp.Constant(999),
                                  "comfort_objective_cost": cp.Constant(2.5)},
        )
        boundaries = pd.date_range(start, periods=steps+1, freq="15min")
        paths = [dict(boundaries=boundaries, ac=cp.Constant(np.full(steps, float(e))),
                      water_scenario_index=w, ev_scenario_index=e-1)
                 for w in range(7) for e in range(1, 8)]
        ev_plan = dict(scenario_plans=paths,
                       scenario_shape=(7, 7),
                       objective_terms={"electricity": cp.Constant(3.),
                                        "comfort": cp.Constant(4.),
                                        "health_soc": cp.Constant(5.),
                                        "terminal": cp.Constant(6.)})
        ev_problem = cp.Problem(cp.Minimize(0), [])
        problem = controller.optimization_problem_milp(
            water, hvac, ev_problem, ev_plan, np.full(steps, .5),
        )
        self.assertTrue(problem.is_mixed_integer())
        self.assertEqual(controller._modeled_home_power_scenarios_kw.shape, (49, steps))
        self.assertEqual(controller._peak_objective_active.sum(), 16)
        self.assertEqual(problem.constraints[-1].shape, (49, 16))
        problem.solve(solver="SCIPY")
        self.assertEqual(problem.status, cp.OPTIMAL)
        self.assertAlmostEqual(controller._objective_components["modeled_home_peak_objective_cost"].value,
                               10 * (7 + 2 + 7 + .5))
        np.testing.assert_allclose(controller._modeled_home_power_scenarios_kw.value[0], 1 + 2 + 1 + .5)
        np.testing.assert_allclose(controller._modeled_home_power_scenarios_kw.value[7], 2 + 2 + 1 + .5)
        self.assertAlmostEqual(controller._objective_components["electricity_objective_cost"].value,
                               .25 * .15 * (4 + 2))
        self.assertNotIn("hpwh_electricity_objective_cost", controller._objective_components)
        self.assertAlmostEqual(sum(float(term.value) for term in controller._objective_components.values()),
                               problem.value)

        controller = self.make_controller(peak_weight=0)
        without_peak = controller.optimization_problem_milp(
            water, hvac, ev_problem, ev_plan, np.full(steps, .5),
        )
        self.assertFalse(controller._peak_objective_active.any())
        self.assertEqual(
            len(without_peak.constraints),
            len(water.problem.constraints) + len(hvac.problem.constraints) + len(ev_problem.constraints),
        )
        self.assertEqual(controller._objective_components["modeled_home_peak_objective_cost"].value, 0)

    def test_five_path_split_hub_threshold_uses_worst_exceedance_and_dt_over_n(self):
        controller = self.make_controller()
        controller.peak_limit_kw = 15.0
        controller.peak_window = "evening"
        self.assertFalse(controller._peak_mask(pd.Timestamp("2025-12-01 00:00"), 32).any())
        start = pd.Timestamp("2025-12-01 15:45")
        steps = 32
        water = SimpleNamespace(
            electric_power_kw=cp.Constant(np.repeat(np.arange(1., 8.)[:, None], steps, axis=1)),
            prices=np.full(steps, .15),
            problem=cp.Problem(cp.Minimize(0), [cp.Variable(boolean=True) == 1]),
            objective_components={},
        )
        hvac = SimpleNamespace(
            electric_power_kw=cp.Constant(np.full(steps, 2.0)),
            prices=np.full(steps, .15), timestamp=start,
            problem=cp.Problem(cp.Minimize(0), []), objective_components={},
        )
        boundaries = pd.date_range(start, periods=steps + 1, freq="15min")
        paths = [dict(boundaries=boundaries, ac=cp.Constant(np.full(steps, float(e + 1))),
                      water_scenario_index=w, ev_scenario_index=e)
                 for w in range(7) for e in range(5)]
        ev_plan = dict(scenario_plans=paths, scenario_shape=(7, 5), objective_terms={})
        problem = controller.optimization_problem_milp(
            water, hvac, cp.Problem(cp.Minimize(0), []), ev_plan,
            hub_other_kw=np.full(steps, 3.0), outside_hub_kw=np.full(steps, 4.0),
        )
        self.assertEqual(controller._modeled_home_power_scenarios_kw.shape, (35, steps))
        self.assertEqual(controller._peak_objective_active.sum(), 24)
        self.assertEqual(problem.constraints[-1].shape, (35, 24))
        problem.solve(solver="SCIPY")
        self.assertEqual(problem.status, cp.OPTIMAL)
        np.testing.assert_allclose(controller._rest_of_home_forecast_kw, np.full(steps, 7.0))
        self.assertAlmostEqual(
            controller._objective_components["modeled_home_peak_objective_cost"].value,
            10.0 * (0.25 / 32) * (7.0 + 2.0 + 5.0 + 3.0 + 4.0 - 15.0),
        )

    def test_joint_ev_discharge_follows_each_water_load_and_inverter_cap(self):
        """One discharge command has 49 load-following rates, not the old reference rate."""
        ev = vehicle(formulation="milp", solver="SCIPY", horizon_hours=.5,
                     health_soc_min=.2, health_soc_max=.8, health_soc_weight=0,
                     comfort_weight=0, enforce_terminal_soc=False)
        controller = EVMPCController(ev)
        mobility = schedule()
        now = mobility.start
        away = schedule([(15, 30, 1)])
        mobilities = (mobility,) * 6 + (away,)
        # The old non-V2H reference is zero; coupled thermal load alone must
        # determine whether V2H discharge is available to the joint model.
        forecasts = tuple(forecast(ev, path, load=0) for path in mobilities)
        scenarios = EVScenarioSet(now, mobilities,
                                  tuple(f.backed_load_kw for f in forecasts), forecasts)
        water_kw = np.array([0, .25, 1, 2, 3, 4, 5], dtype=float)
        # All paths share the observed first command; water uncertainty starts
        # at the next control interval, where rates can recourse by water path.
        water = cp.Constant(np.column_stack([np.ones(7), water_kw]))
        hp_and_fan = cp.Constant(np.full(2, 1.5))
        built, plan = build_joint_ev_scenario_problem(
            controller, now, .8, scenarios, water, hp_and_fan,
            pd.Timedelta(minutes=15), eligible_power_upper_kw=6.5,
        )
        self.assertEqual(plan["scenario_shape"], (7, 7))
        self.assertEqual(len(plan["scenario_plans"]), 49)
        charge, discharge = plan["shared_first_command"]
        problem = cp.Problem(built.objective, [*built.constraints, discharge == 1,
            *(p["discharge_mode"][1] == 1 for p in plan["scenario_plans"]
              if p["ev_scenario_index"] < 6)])
        problem.solve(solver="SCIPY")
        self.assertEqual(problem.status, cp.OPTIMAL)
        self.assertLess(float(charge.value), .5)
        for w, draw in enumerate(water_kw):
            expected = min(1.5 + draw, ev.hems.max_discharge_kw)
            for e in range(7):
                path = plan["scenario_plans"][7 * w + e]
                actual_discharge = expected if e < 6 else 0.
                self.assertEqual((path["water_scenario_index"], path["ev_scenario_index"]), (w, e))
                self.assertGreater(float(path["discharge_mode"].value[0]), .5)
                self.assertAlmostEqual(float(path["discharge"].value[0]), 2.5, places=5)
                self.assertAlmostEqual(float(path["discharge"].value[1]), actual_discharge, places=5)
                self.assertAlmostEqual(float(path["ac"].value[1]), -actual_discharge, places=5)
                if e == 6:
                    # The scenario trip starts at this boundary, before the
                    # away interval and before its return at horizon end.
                    departure_index = path["horizon"].t.get_loc(now + pd.Timedelta(minutes=15))
                    self.assertEqual(path["horizon"].trip_timing, "departure")
                    self.assertAlmostEqual(float(path["state"].value[departure_index]),
                                           .8 - (.25 * 2.5 / ev.hems.discharge_efficiency
                                                 + path["horizon"].trip_kwh.sum()) / ev.capacity_kwh,
                                           places=5)
                self.assertAlmostEqual(float(path["state"].value[2]),
                                       .8 - (.25 * (2.5 + actual_discharge) /
                                             ev.hems.discharge_efficiency +
                                             path["horizon"].trip_kwh.sum()) / ev.capacity_kwh,
                                       places=5)
        for name, total in plan["objective_terms"].items():
            path_costs = [p["objective_terms"][name].value for p in plan["scenario_plans"]]
            self.assertAlmostEqual(float(total.value),
                                   max(path_costs) if name == "comfort" else np.mean(path_costs))

    def test_requires_approved_eight_hour_ev_horizon(self):
        controller = self.make_controller()
        controller.ev_controller.horizon = pd.Timedelta(hours=12)
        with self.assertRaisesRegex(ValueError, "eight hours"):
            controller.__post_init__()

    def test_ev_observation_uses_previous_capacity_before_soh_refresh(self):
        controller = self.make_controller()
        ev = controller.ev_controller
        ev.equipment_name = "EV_01A"
        ev.ev = SimpleNamespace(capacity_kwh=60.)
        ev.current_capacity = 50.
        ev.current_soh = 50. / 60.
        ev.soc_history = [.4]
        observed_capacity = []
        ev._observe = lambda *_: observed_capacity.append(ev.current_capacity)
        timestamp = pd.Timestamp("2018-01-03 03:00")
        data = pd.DataFrame(index=pd.date_range(timestamp, periods=2, freq="min"))
        forecast = SimpleNamespace(data=data, ev_states={
            "EV_01A SOC (-)": .4, "EV_01A SOH (-)": .8,
            "EV_01A Capacity (kWh)": 48.,
        })
        soc, step = controller._observe_ev(timestamp, {"EV_01A Electric Power (kW)": -2.}, forecast)
        self.assertEqual((soc, step), (.4, pd.Timedelta(minutes=1)))
        self.assertEqual(observed_capacity, [50.])
        self.assertEqual(ev.current_capacity, 48.)

    def test_one_solver_call_dispatches_all_three_devices(self):
        controller = self.make_controller()
        timestamp = pd.Timestamp("2018-01-03 03:00")
        water = SimpleNamespace(heat=cp.Constant(1),
                                electric_power_kw=cp.Constant(np.ones((7, 32))))
        hvac = SimpleNamespace(heat_hvac=cp.Constant(1), heat_hp=cp.Constant(1),
                               heat_er=cp.Constant(np.full(32, 6000.)),
                               electric_power_kw=cp.Constant(np.full(32, 7.5)))
        controller.water_heater_controller.prepare_milp_scenario_step = lambda *_: water
        controller.water_heater_controller.apply_milp_scenario_solution = lambda *_: {"Water Heating": {"Setpoint": 60}}
        controller.hvac_controller.prepare_mixed_int_step = lambda *_: hvac
        controller.hvac_controller.apply_mixed_int_solution = lambda *_: {"HVAC Heating": {"Setpoint": 21}}
        controller.hvac_controller._time_limited_solution_is_feasible = lambda *_: True
        controller.hvac_controller._solver_metrics = lambda *_: {"relative_gap": .04, "time_limit_hit": False}
        controller.ev_controller.equipment_name = "EV_01A"
        controller._eligible_power_upper_kw = lambda *_: 10.
        ev_plan = dict(scenario_plans=[{"state": cp.Constant(1)} for _ in range(49)])
        rest = pd.Series(np.ones(32), index=pd.date_range(timestamp, periods=32, freq="15min"))
        provider = SimpleNamespace(forecast_kw=lambda *_a, **_k: rest, method_label="test")
        forecast = SimpleNamespace(rest_of_home_forecast=provider,
                                   for_device=lambda *_: SimpleNamespace(forecasts=[object()] * 7))

        class FakeProblem:
            status = cp.OPTIMAL
            value = 1.0

            def __init__(self):
                self.calls = []

            def solve(self, **kwargs):
                self.calls.append(kwargs)

        problem = FakeProblem()
        controller._objective_components = {"test_cost": cp.Constant(1)}
        with (patch.object(controller, "_observe_ev", return_value=(.4, pd.Timedelta(minutes=1))),
              patch("examples.single_home_hems.single_home_hems.controllers.whole_home_joint."
                    "build_joint_ev_scenario_problem", return_value=(object(), ev_plan)) as builder,
              patch.object(controller, "optimization_problem_milp", return_value=problem),
              patch.object(controller, "_record_ev_solution"),
              patch.object(controller, "_record_plan"),
              patch.object(controller, "_ev_control", return_value={"EV_01A": {"P Setpoint": 0}})):
            # The provider result only needs to be recognized as EVScenarioSet.
            from examples.single_home_hems.single_home_hems.control_forecasts import EVScenarioSet
            scenario = EVScenarioSet(timestamp, tuple([object()] * 7),
                                     tuple([rest] * 7), tuple([object()] * 7))
            forecast.for_device = lambda *_: scenario
            commands = controller.controls_for_step(timestamp, {}, {}, forecast)
        predicted_hp_and_fan_kw = builder.call_args.args[5]
        np.testing.assert_allclose(predicted_hp_and_fan_kw.value, np.full(32, 1.5))
        plant_backed_kw = backed_power({
            "HVAC Heating Main Power (kW)": 1.,
            "HVAC Heating Fan Power (kW)": .5,
            "HVAC Cooling Main Power (kW)": 0.,
            "HVAC Cooling Fan Power (kW)": 0.,
            "Water Heating Electric Power (kW)": 1.,
        })
        self.assertAlmostEqual(float(predicted_hp_and_fan_kw.value[0] +
                                     water.electric_power_kw.value[0, 0]), plant_backed_kw)
        self.assertEqual(len(problem.calls), 1)
        self.assertEqual(problem.calls[0]["solver"], "GUROBI")
        self.assertEqual(problem.calls[0]["TimeLimit"], 300)
        self.assertEqual(problem.calls[0]["MIPGap"], .05)
        self.assertEqual(set(commands), {"Water Heating", "HVAC Heating", "EV_01A"})
        self.assertEqual(len(controller.solve_history), 1)

    def test_transient_wls_dns_error_retries_same_problem(self):
        controller = self.make_controller()

        class FakeProblem:
            calls = 0

            def solve(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise gp.GurobiError(10003, "Could not resolve host: token.gurobi.com")

        problem = FakeProblem()
        with (patch.object(controller, "_refresh_gurobi_environment") as refresh,
              patch("examples.single_home_hems.single_home_hems.controllers.whole_home_joint.time.sleep")):
            controller._solve_problem_with_retries(problem)
        self.assertEqual(problem.calls, 2)
        self.assertEqual(refresh.call_count, 1)
        self.assertEqual(controller._wls_network_retry_count, 1)
        self.assertEqual(controller._solver_retry_count, 0)

    def test_joint_ev_dispatch_uses_solved_rate_when_reference_is_lower(self):
        controller = self.make_controller()
        ev = controller.ev_controller
        timestamp = pd.Timestamp("2018-01-03 03:00")
        ev.ev = SimpleNamespace(vehicle_id="01A", charger_power_kw=4.)
        ev.equipment_name = "EV_01A"
        ev.fallback = False
        ev.next_solve = timestamp
        ev.target = -3.25
        ev.action_history = []
        projected = []
        ev._project = lambda _t, target, _soc, _available, backed: (
            projected.append((target, backed)) or -1, target, False)
        mobility = SimpleNamespace(available=lambda *_: [True])
        forecast_data = SimpleNamespace(
            ev_schedules={"01A": mobility},
            backed_load_kw=pd.Series([.5], index=[timestamp]),
        )
        command = controller._ev_control(timestamp, .6, pd.Timedelta(minutes=1), forecast_data)
        self.assertEqual(projected, [(-3.25, 3.25)])
        self.assertEqual(command["EV_01A"], {"P Setpoint": -3.25, "Backed Load (kW)": 3.25})


class WholeHomeJointWeatherCliTests(unittest.TestCase):
    def test_predict_weather_reaches_controlled_case_only(self) -> None:
        self.assertEqual(
            run_whole_home_joint._validated_args([]).weather_forecast_mode,
            "perfect",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "forecast.parquet"
            run = pd.Timestamp("2018-01-01T00:00:00Z")
            pd.DataFrame([{
                "run_time_utc": run,
                "valid_time_utc": valid,
                "valid_time_local": valid.tz_convert("America/Indiana/Indianapolis"),
                "temp_air": 1.0,
                "relative_humidity": 0.5,
                "ghi": 0.0,
                "dni": 0.0,
                "dhi": 0.0,
            } for valid in (run + pd.Timedelta(hours=hour) for hour in range(1, 25))]).to_parquet(path)
            index = pd.date_range("2018-01-01 03:00", periods=9 * 60, freq="min")
            twin = Mock()
            twin._run_stepwise.return_value = (pd.DataFrame(index=index),)
            twin.raw_directory = root / "twin_raw"
            calls = []

            class StopAtControlledCase(Exception):
                pass

            def mock_simulation(*args, **kwargs):
                calls.append(kwargs)
                if len(calls) == 1:
                    return twin
                raise StopAtControlledCase

            with (
                patch.object(
                    run_whole_home_joint, "SingleHomeHEMSSimulation",
                    side_effect=mock_simulation,
                ),
                patch.object(
                    run_whole_home_joint, "build_ev_schedules",
                    side_effect=lambda _config, home: {
                        ev.vehicle_id: SimpleNamespace(metadata={}) for ev in home.evs
                    },
                ),
                patch.object(
                    run_whole_home_joint, "backed_power",
                    side_effect=lambda raw, _loads: np.zeros(len(raw)),
                ),
                patch.object(
                    run_whole_home_joint.RestOfHomeHybridForecast,
                    "from_csv", return_value=SimpleNamespace(method_label="test", metadata={}),
                ),
                redirect_stdout(io.StringIO()),
            ):
                with self.assertRaises(StopAtControlledCase):
                    run_whole_home_joint.main([
                        "--output-directory", str(root / "out"),
                        "--weather-forecast-mode", "predict",
                        "--weather-forecast-file", str(path),
                    ])
            self.assertEqual(len(calls), 2)
            self.assertNotIn("weather_archive", calls[0])
            self.assertEqual(len(calls[1]["weather_archive"].data), 24)


if __name__ == "__main__":
    unittest.main()
