"""Shared calibration, exact rainflow cuts, certified solves and baseline economics."""
from dataclasses import replace
from itertools import product
import json
from pathlib import Path
import time
import unittest
from unittest.mock import patch

import cvxpy as cp
import numpy as np
import pandas as pd

from test.test_ev_v2h import CONFIG, ROOT, vehicle, schedule, forecast
from test import test_ev_v2h as plant_fixture
from examples.five_home_block.five_home_block.ev_degradation import BatteryDegradation, rainflow_cycles
from examples.single_home_hems.single_home_hems.controllers.ev import EVMPCController, master_lower_bound
from examples.single_home_hems.single_home_hems.controllers.solver import solve_ocp
from examples.single_home_hems.single_home_hems.control_config import load_unified
from examples.single_home_hems.single_home_hems.ev_metrics import ev_metrics
from examples.single_home_hems.single_home_hems.unified import baseline_economics


class RainflowModelTests(unittest.TestCase):
    def battery(self, **options):
        ev = vehicle(**options)
        return BatteryDegradation(ev.capacity_kwh, ev.hems)

    def test_reference_full_cycle_matches_all_three_models(self):
        b = self.battery()
        self.assertAlmostEqual(b.reference_cycle_life, 4500)
        self.assertAlmostEqual(b.reference_loss(.8, .6), .2/3000)
        for key, value in b.costs([.2, 1., .2]).items():
            if "cost" in key:
                self.assertAlmostEqual(value, b.depreciable_usd/b.reference_cycle_life)
        for key in ("rainflow", "linear", "reference"):
            self.assertAlmostEqual(b.costs([.2, 1.])[key+"_degradation_cost_usd"],
                                   .5*b.costs([.2, 1., .2])[key+"_degradation_cost_usd"])

    def test_nested_cycle_and_open_residue_are_not_periodically_closed(self):
        x = [.1, .8, .4, .6, .2, .9]
        pairs = rainflow_cycles(x)
        counted = sorted((round(abs(x[a]-x[b]), 6), n) for a, b, n in pairs)
        self.assertEqual(counted, [(.2, 1), (.6, 1), (.8, .5)])
        self.assertEqual(self.battery().costs([.5]*5)["reference_degradation_cost_usd"], 0)

    def test_signed_mean_soc_stress_affects_only_reference_cost(self):
        b = self.battery()
        lower, centered, shifted = (b.costs(x) for x in ([.3, .5, .3], [.5, .7, .5], [.7, .9, .7]))
        for key in ("rainflow", "linear"):
            self.assertAlmostEqual(lower[key+"_degradation_cost_usd"], centered[key+"_degradation_cost_usd"])
            self.assertAlmostEqual(centered[key+"_degradation_cost_usd"], shifted[key+"_degradation_cost_usd"])
        self.assertLess(lower["reference_degradation_cost_usd"], centered["reference_degradation_cost_usd"])
        self.assertGreater(shifted["reference_degradation_cost_usd"], centered["reference_degradation_cost_usd"])
        self.assertAlmostEqual(shifted["reference_degradation_cost_usd"]/centered["reference_degradation_cost_usd"],
                               np.exp(b.options.soc_stress_coefficient*.2))

    def test_supporting_planes_with_plateaus_ties_and_small_perturbations(self):
        b, rng = self.battery(), np.random.default_rng(14)
        for i in range(1000):
            x = rng.uniform(size=10)
            if i % 2:
                x = np.round(x, 1)
            y = np.clip(x+rng.normal(0, .0001, 10), 0, 1) if i % 3 else rng.uniform(size=10)
            fx, g = b.rainflow_cost_gradient(x)
            fy, _ = b.rainflow_cost_gradient(y)
            self.assertLessEqual(fx+g@(y-x), fy+1e-10)

    def test_history_increment_telescopes_without_resetting_open_cycles(self):
        b, path = self.battery(), [.5, .8, .4, .7, .2, .9, .3]
        total = sum(b.costs(path[:k+1])["rainflow_degradation_cost_usd"]-
                    b.costs(path[:k])["rainflow_degradation_cost_usd"] for k in range(1, len(path)))
        self.assertAlmostEqual(total, b.costs(path)["rainflow_degradation_cost_usd"])

    def test_shared_parameter_scaling_and_invalid_calibration(self):
        base, doubled = self.battery(), self.battery(battery_replacement_cost_per_kwh=238)
        self.assertAlmostEqual(base.replacement_usd/base.capacity_kwh, 119)
        self.assertAlmostEqual(base.residual_usd/base.capacity_kwh, 0)
        self.assertAlmostEqual(base.depreciable_usd/base.capacity_kwh, 119)
        for key in ("rainflow", "linear", "reference"):
            key += "_degradation_cost_usd"
            self.assertAlmostEqual(doubled.costs([.2, .8, .2])[key], 2*base.costs([.2, .8, .2])[key])
        for options in (dict(rainflow_exponent=.9), dict(reference_mean_soc=.1),
                        dict(reference_cycle_count=0), dict(end_of_life_soh=1),
                        dict(calibration_end_of_life_soh=1), dict(calibration_end_of_life_soh=float("nan")),
                        dict(battery_residual_value_per_kwh=210), dict(soc_stress_coefficient=-1),
                        dict(rainflow_max_iterations=0), dict(degradation_method="unknown")):
            with self.subTest(options=options), self.assertRaises(ValueError):
                vehicle(**options)

    def test_residual_equal_to_new_cost_zeroes_every_aging_cost(self):
        b = self.battery(battery_residual_value_per_kwh=119)
        for key, value in b.costs([.2, 1., .2]).items():
            if "cost" in key:
                self.assertEqual(value, 0)

    def test_economic_endpoint_changes_cost_not_physical_fade(self):
        original = self.battery(end_of_life_soh=.8)
        extended = self.battery(end_of_life_soh=.7)
        self.assertAlmostEqual(original.reference_cycle_life, 3000)
        self.assertAlmostEqual(extended.reference_cycle_life, 4500)
        for depth, mean in ((.8, .6), (.2, .4), (.3, .7)):
            self.assertAlmostEqual(original.reference_loss(depth, mean), extended.reference_loss(depth, mean))
        for key in ("rainflow", "linear", "reference"):
            name = key+"_degradation_cost_usd"
            self.assertAlmostEqual(extended.costs([.2, 1., .2])[name],
                                   original.costs([.2, 1., .2])[name]*2/3)
            self.assertAlmostEqual(4500*extended.costs([.2, 1., .2])[name], extended.depreciable_usd)


class DegradationSolverTests(unittest.TestCase):
    def controller(self, kind="convex", solver="CLARABEL", method="rainflow", **options):
        ev = vehicle(formulation=kind, solver=solver, enable_degradation=True,
                     degradation_method=method, **options)
        c, m = EVMPCController(ev), schedule()
        f = forecast(ev, m)
        return c, m, f

    def test_both_methods_and_all_backend_formulations(self):
        pairs = [("convex", "CLARABEL"), ("convex", "SCIPY"), ("milp", "SCIPY")]
        if "GUROBI" in cp.installed_solvers():
            pairs += [("convex", "GUROBI"), ("milp", "GUROBI")]
        for method, (kind, solver) in product(("rainflow", "linear_throughput"), pairs):
            with self.subTest(method=method, kind=kind, solver=solver):
                c, m, f = self.controller(kind, solver, method)
                c.controls_for_step(m.start, {c.equipment_name+" SOC (-)": .5, c.equipment_name+" SOH (-)": 1.}, {}, f)
                self.assertFalse(c.fallback, c.solve_history)
                plan = c.plans[-1]
                self.assertAlmostEqual(plan["objective"], sum(plan["objective_terms"].values()), places=6)
                if method == "rainflow":
                    s = c.solve_history[-1]
                    self.assertTrue(s["rainflow_converged"])
                    self.assertGreaterEqual(s["rainflow_iterations"], 2)
                    self.assertLessEqual(s["rainflow_lower_bound"], s["rainflow_upper_bound"]+1e-7)

    def test_milp_exact_objective_matches_exhaustive_mode_enumeration(self):
        c, m, f = self.controller("milp", "SCIPY", horizon_hours=.75,
                                  health_weight=20, rainflow_absolute_tolerance=1e-8, mip_relative_gap=.001)
        c.controls_for_step(m.start, {c.equipment_name+" SOC (-)": .5, c.equipment_name+" SOH (-)": 1.}, {}, f)
        b = BatteryDegradation(10, c.options)
        candidates = []
        for ac in product((-3., 0., 4.), repeat=3):
            ac = np.array(ac)
            chemical = np.where(ac >= 0, ac*.9, ac/.8)
            soc = np.r_[.5, .5+np.cumsum(.25*chemical/10)]
            if np.all((soc >= 0) & (soc <= 1)):
                candidates.append(.25*np.sum(3+ac)/3+20*b.costs(soc)["rainflow_degradation_cost_usd"]/3)
        self.assertFalse(c.fallback)
        self.assertAlmostEqual(c.plans[-1]["objective"], min(candidates), places=6)

    def test_history_changes_increment_without_charging_past_damage_again(self):
        c, m, f = self.controller()
        c.soc_history = [.5, .9, .2, .5]
        c.controls_for_step(m.start, {c.equipment_name+" SOC (-)": .5, c.equipment_name+" SOH (-)": 1.}, {}, f)
        b = BatteryDegradation(10, c.options)
        past = b.costs(c.soc_history)["rainflow_degradation_cost_usd"]
        planned = b.costs(c.soc_history[:-1]+c.plans[-1]["soc"])["rainflow_degradation_cost_usd"]
        self.assertAlmostEqual(c.plans[-1]["objective_terms"]["degradation"], (planned-past)/4)
        self.assertAlmostEqual(c.solve_history[-1]["historical_rainflow_cost_usd"], past)

    def test_convex_exact_solution_matches_one_step_analytic_optimum(self):
        # Keep an interior optimum under the new, lower net depreciable value.
        weight = 1000
        c, m, f = self.controller(horizon_hours=.25, health_weight=weight,
                                  rainflow_relative_tolerance=1e-6, rainflow_absolute_tolerance=1e-8,
                                  solver_options={"tol_gap_abs": 1e-10, "tol_gap_rel": 1e-10,
                                                  "tol_feas": 1e-10})
        c.controls_for_step(m.start, {c.equipment_name+" SOC (-)": .5, c.equipment_name+" SOH (-)": 1.}, {}, f)
        b = BatteryDegradation(10, c.options)
        beta, dt = c.options.rainflow_exponent, .25
        # For chemical discharge q>=0: dt*(3-.8*q) + weight*R*a/2*(dt*q/E)^beta.
        denominator = weight*b.depreciable_usd*b.power_stress_coefficient/2*beta*(dt/10)**beta
        q = min(3/.8, (dt*.8/denominator)**(1/(beta-1)))
        expected = dt*(3-.8*q)+weight*b.costs([.5, .5-dt*q/10])["rainflow_degradation_cost_usd"]
        self.assertFalse(c.fallback, c.solve_history)
        self.assertAlmostEqual(c.plans[-1]["objective"], expected, places=6)
        self.assertGreater(c.solve_history[-1]["rainflow_iterations"], 2)

    def test_missing_certificate_never_accepts_an_optimal_native_master(self):
        c, m, f = self.controller(rainflow_max_iterations=3)
        with patch("examples.single_home_hems.single_home_hems.controllers.ev.master_lower_bound", return_value=-np.inf):
            c.controls_for_step(m.start, {c.equipment_name+" SOC (-)": .5, c.equipment_name+" SOH (-)": 1.}, {}, f)
        self.assertTrue(c.fallback)
        self.assertFalse(c.solve_history[-1]["rainflow_converged"])

    def test_milp_objective_gap_cannot_hide_missing_rainflow_cost(self):
        c, m, f = self.controller(kind="milp", solver="SCIPY", horizon_hours=.5)
        p, plan = c.build_problem(m.start, .5, f)
        # A large unavoidable comfort cost makes the first master's overall
        # relative gap tiny, but must not permit a zero aging-cost envelope.
        plan["objective_terms"]["comfort"] += cp.Constant(1e6)
        p, accepted = c._solve_rainflow(p, plan, {}, time.perf_counter()+30)
        info = p.hems_diagnostic
        self.assertTrue(accepted, info)
        self.assertGreater(info["rainflow_cuts"], 0)
        self.assertTrue(info["rainflow_model_certified"])
        self.assertLessEqual(info["rainflow_approximation_error_usd"], info["rainflow_model_tolerance_usd"])

    def test_disabled_or_zero_weight_omits_rainflow_master_variable(self):
        for options in (dict(enable_degradation=False), dict(health_weight=0),
                        dict(battery_replacement_cost_per_kwh=0, battery_residual_value_per_kwh=0),
                        dict(battery_residual_value_per_kwh=119)):
            ev = vehicle(**({"enable_degradation": True} | options))
            c, m = EVMPCController(ev), schedule()
            _, q = c.build_problem(m.start, .5, forecast(ev, m))
            self.assertIsNone(q["rainflow_theta"])
            self.assertEqual(q["degradation"].value, 0)

    def test_iteration_limit_falls_back_even_with_optimal_master(self):
        c, m, f = self.controller(rainflow_max_iterations=1)
        c.controls_for_step(m.start, {c.equipment_name+" SOC (-)": .5, c.equipment_name+" SOH (-)": 1.}, {}, f)
        self.assertTrue(c.fallback)
        self.assertEqual(c.solve_history[-1]["status"], "rainflow_not_converged")
        self.assertEqual(c.plans, [])

    def test_expired_budget_does_not_start_another_backend(self):
        c, m, f = self.controller()
        p, q = c.build_problem(m.start, .5, f)
        with patch("examples.single_home_hems.single_home_hems.controllers.ev.solve_ocp") as solve:
            p, accepted = c._solve_rainflow(p, q, {}, time.perf_counter()-1)
        solve.assert_not_called()
        self.assertFalse(accepted)
        self.assertTrue(p.hems_diagnostic["timed_out"])

    def test_each_backend_receives_remaining_budget(self):
        for solver in ("SCIPY", "CLARABEL", "GUROBI"):
            p = cp.Problem(cp.Minimize(0))
            with patch.object(p, "solve") as solve:
                solve_ocp(p, solver, time_limit_seconds=7)
            kw = solve.call_args.kwargs
            limit = kw["scipy_options"]["time_limit"] if solver == "SCIPY" else kw["TimeLimit" if solver == "GUROBI" else "time_limit"]
            self.assertEqual(limit, 7)

    def test_scipy_lp_ignores_dummy_mip_bound_and_preserves_objective_offset(self):
        for boolean in (False, True):
            x = cp.Variable(boolean=boolean)
            p = cp.Problem(cp.Minimize(7+x), [x >= 0, x <= 1])
            solve_ocp(p, "SCIPY")
            self.assertAlmostEqual(master_lower_bound(p, "SCIPY"), 7, places=6)

    def test_linear_throughput_normalization_and_driving(self):
        c, m, _ = self.controller(method="linear_throughput")
        m = schedule([(15, 30, 5)])
        p, q = c.build_problem(m.start, .5, forecast(c.ev, m))
        solve_ocp(p, "CLARABEL")
        expected = q["battery"].throughput_usd_per_kwh*(np.sum(q["horizon"].dt*np.abs(q["chemical"].value))+1)/4
        self.assertAlmostEqual(q["degradation"].value, expected)


class DegradationReportingTests(unittest.TestCase):
    def test_departure_metrics_exclude_warmup_cooldown_and_use_predeparture_soc(self):
        ev = vehicle(comfort_soc_min=.4)
        m = schedule([(0, 5, 0), (15, 20, 0), (30, 40, 0)])
        index = pd.date_range(m.start+pd.Timedelta(minutes=10), periods=20, freq="min")
        key = "EV_"+ev.vehicle_id
        raw = pd.DataFrame({key+" SOC (-)": .2, key+" End SOC (-)": .1,
                            key+" Electric Power (kW)": 0., key+" Unmet Load (kWh)": 0.,
                            key+" Safety Override": 0}, index=index)
        row = ev_metrics(raw, replace(CONFIG.homes[0], evs=(ev,)), pd.Series(1., index=index),
                         1/60, mobility={ev.vehicle_id: m}).iloc[0]
        self.assertEqual(row.departure_count, 1)
        self.assertEqual(row.departure_comfort_shortfall_events, 1)
        self.assertAlmostEqual(row.mean_departure_comfort_shortfall, .2)
        self.assertAlmostEqual(row.max_departure_comfort_shortfall, .2)

    def test_net_savings_includes_baseline_reference_cost(self):
        rows = pd.DataFrame(dict(home_id="h", device="whole_home", case=case, cost_usd=cost,
                                 reference_degradation_cost_usd=age) for case, cost, age in
                            [("all_baseline", 10, 2), ("non_v2h_twin", 8, 3), ("advanced", 6, 5)])
        result = baseline_economics(rows)
        self.assertEqual(result.operating_cost_savings_usd.tolist(), [0, 2, 4])
        self.assertEqual(result.degradation_cost_savings_usd.tolist(), [0, -1, -3])
        self.assertEqual(result.net_savings_usd.tolist(), [0, 1, 1])
        self.assertNotIn("net_profit_usd", result)
        self.assertFalse(any("twin" in name for name in result.columns))

    def test_realized_metrics_preserve_charge_then_trip_extrema(self):
        ev = vehicle(degradation_method="linear_throughput")
        key = "EV_"+ev.vehicle_id
        raw = pd.DataFrame({key+" SOC (-)": [.5, .4], key+" End SOC (-)": [.6, .4],
                            key+" Electric Power (kW)": [4, 0], key+" Unmet Load (kWh)": [0, 8],
                            key+" Safety Override": [0, 0]})
        row = ev_metrics(raw, replace(CONFIG.homes[0], evs=(ev,)), pd.Series([1, 1]), 1/60).iloc[0]
        self.assertAlmostEqual(row.battery_throughput_kwh, 3)
        self.assertEqual(row.estimated_degradation_cost_usd, row.linear_degradation_cost_usd)
        self.assertGreater(row.reference_degradation_cost_usd, 0)

    def test_baseline_stops_at_comfort_not_health_minimum(self):
        fixture = plant_fixture.EVPlantTests()
        plant = fixture.plant(replace(vehicle(comfort_soc_min=.55, health_soc_min=.1), initial_soc_fraction=.549))
        fixture.step(plant, None)
        self.assertAlmostEqual(plant.soc, .55)
        fixture.step(plant, None)
        self.assertEqual(plant.electric_kw, 0)

    def test_unified_documents_require_control_targets_and_battery_calibration(self):
        directory = ROOT/"examples/single_home_hems/config"
        home = json.loads((directory/"unified_home.json").read_text())
        control = json.loads((directory/"unified_control.json").read_text())
        calibration_missing = json.loads(json.dumps(home))
        del calibration_missing["homes"]["home_01"]["evs"]["01A"]["physical"]["calibration_end_of_life_soh"]
        with patch.object(Path, "read_text", side_effect=[json.dumps(calibration_missing), json.dumps(control)]):
            with self.assertRaisesRegex(ValueError, "calibration_end_of_life_soh"):
                load_unified("h.json", "c.json", ROOT)
        del home["homes"]["home_01"]["evs"]["01A"]["physical"]["reference_cycle_depth"]
        with patch.object(Path, "read_text", side_effect=[json.dumps(home), json.dumps(control)]):
            with self.assertRaisesRegex(ValueError, "reference_cycle_depth"):
                load_unified("h.json", "c.json", ROOT)
        home = json.loads((directory/"unified_home.json").read_text())
        del control["defaults"]["ev"]["comfort_soc_min"]
        del control["homes"]["home_01"]["evs"]["01A"]["comfort_soc_min"]
        with patch.object(Path, "read_text", side_effect=[json.dumps(home), json.dumps(control)]):
            with self.assertRaisesRegex(ValueError, "comfort_soc_min"):
                load_unified("h.json", "c.json", ROOT)
