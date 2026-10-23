"""Scenario EV selection, persistence, two-stage OCP and solver regressions."""
from dataclasses import replace
from contextlib import redirect_stdout
import io
from types import SimpleNamespace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import cvxpy as cp
import numpy as np
import pandas as pd

from test.test_ev_v2h import ROOT, CONFIG, vehicle, schedule, forecast
from examples.single_home_hems.single_home_hems.control_config import resolve_control, load_unified
from examples.single_home_hems.single_home_hems.control_config import UnifiedConfiguration
from examples.single_home_hems.single_home_hems.unified import run_home, run_unified, CASES
from examples.single_home_hems.single_home_hems.control_forecasts import EVScenarioSet
from examples.single_home_hems.single_home_hems.ev_scenarios import (
    ScenarioEVForecast, NoEVScenarios, persistent_prefix, FILTERS)
from examples.single_home_hems.single_home_hems.controllers.ev import EVMPCController
from examples.single_home_hems.single_home_hems.controllers.solver import solve_ocp, usable_solution
from examples.five_home_block.five_home_block.ev_schedule import EVSchedule


def historical(trips=True):
    now = schedule().start
    rows = [dict(departure=now+pd.Timedelta(days=d, hours=8),
                 arrival=now+pd.Timedelta(days=d, hours=10), distance_km=10+d % 5)
            for d in range(-40, 3)] if trips else []
    return EVSchedule(pd.DataFrame(rows, columns=["departure", "arrival", "distance_km"]),
                      now-pd.Timedelta(days=40), now+pd.Timedelta(days=3), {})


def scenario_control(**settings):
    return resolve_control({"defaults": {"ev": dict(controller="mpc", horizon_hours=1,
        forecast=dict(mode="scenario", **settings))}}, "h", "ev", "v")


def ensemble(ev, trips=((), ((30, 45, 2),))):
    paths = tuple(schedule(t) for t in trips)
    forecasts = tuple(forecast(ev, p) for p in paths)
    return EVScenarioSet(paths[0].start, paths, tuple(f.backed_load_kw for f in forecasts),
                         forecasts, tuple(str(i) for i in range(len(paths))))


class EVScenarioForecastTests(unittest.TestCase):
    def issue(self, method="knn_state", **settings):
        ev, m = vehicle(), historical()
        f = forecast(ev, schedule())
        f = replace(f, ev_schedules={ev.vehicle_id: m})
        provider = ScenarioEVForecast(ev.vehicle_id, scenario_control(scenario_method=method, **settings))
        return provider, provider.at(schedule().start, f, pd.Timedelta(hours=1)), f

    def test_defaults_completed_nonoverlapping_and_repeatable(self):
        provider, paths, f = self.issue()
        self.assertEqual(provider.settings["scenario_count"], 5)
        self.assertEqual(provider.settings["lookback_days"], 30)
        self.assertEqual(len(paths.paths), 5)
        np.testing.assert_allclose(paths.probabilities, .2)
        origins = sorted(pd.Timestamp(x) for x in paths.candidate_ids)
        now = schedule().start
        self.assertTrue(all(t+pd.Timedelta(hours=1) <= now for t in origins))
        self.assertTrue(all(b-a >= pd.Timedelta(hours=1) for a, b in zip(origins, origins[1:])))
        repeat = provider.at(now, f, pd.Timedelta(hours=1))
        self.assertEqual(paths.candidate_ids, repeat.candidate_ids)
        for p in paths.paths:
            self.assertTrue(p.available([now])[0])
        self.assertEqual(paths.diagnostics["prefix_applied_count"],
                         sum(p.metadata["prefix_applied"] for p in paths.paths))

    def test_all_filter_methods_and_future_data_independence(self):
        for method in FILTERS:
            with self.subTest(method=method):
                _, paths, _ = self.issue(method)
                self.assertGreater(len(paths.paths), 0)
                self.assertTrue(all(pd.Timestamp(t).time() == paths.issued_at.time() for t in paths.candidate_ids))
        provider, paths, f = self.issue()
        m = f.ev_schedules[vehicle().vehicle_id]
        changed = m.trips.copy()
        changed.loc[changed.departure >= paths.issued_at, "distance_km"] = 10000
        altered = replace(f, ev_schedules={vehicle().vehicle_id: replace(m, trips=changed)})
        repeat = provider.at(paths.issued_at, altered, pd.Timedelta(hours=1))
        self.assertEqual(paths.candidate_ids, repeat.candidate_ids)
        for a, b in zip(paths.paths, repeat.paths):
            pd.testing.assert_frame_equal(a.trips, b.trips)

    def test_short_pool_uses_fewer_and_zero_candidates_falls_back(self):
        _, paths, _ = self.issue(lookback_days=1/24, knn_history_hours=1)
        self.assertEqual(len(paths.paths), 1)
        ev = vehicle()
        provider = ScenarioEVForecast(ev.vehicle_id, scenario_control(lookback_days=.01))
        f = replace(forecast(ev, schedule()), ev_schedules={ev.vehicle_id: historical()})
        with self.assertRaises(NoEVScenarios):
            provider.at(schedule().start, f, pd.Timedelta(hours=1))
        self.assertTrue(provider.history[-1]["no_scenarios_selected"])
        self.assertEqual(provider.issued_forecasts[-1][2], ())
        c = EVMPCController(ev)
        c._solve(schedule().start, .5, replace(f, providers={c.equipment_name: provider}))
        self.assertTrue(c.fallback)
        self.assertEqual(c.next_solve, schedule().start+pd.Timedelta(minutes=15))

    def test_requires_full_pool_and_history(self):
        ev = vehicle()
        p = ScenarioEVForecast(ev.vehicle_id, scenario_control())
        with self.assertRaisesRegex(ValueError, "lookback/history"):
            p.at(schedule().start, forecast(ev, schedule()), pd.Timedelta(hours=1))

    def test_nonoverlap_uses_full_forecast_length(self):
        ev = vehicle()
        control = scenario_control()
        control["forecast_hours"] = 2
        provider = ScenarioEVForecast(ev.vehicle_id, control)
        f = replace(forecast(ev, schedule()), ev_schedules={ev.vehicle_id: historical()})
        paths = provider.at(schedule().start, f, pd.Timedelta(hours=1))
        origins = sorted(pd.Timestamp(t) for t in paths.candidate_ids)
        self.assertEqual(paths.diagnostics["forecast_hours"], 2)
        self.assertTrue(all(b-a >= pd.Timedelta(hours=2) for a, b in zip(origins, origins[1:])))

    def test_persistence_home_discards_enclosed_trips_without_compounding(self):
        m = schedule([(-5, 10, 2), (15, 20, 3), (40, 50, 7)])
        p = persistent_prefix(m, m.start, pd.Timedelta(hours=1), pd.Timedelta(minutes=30), True, pd.Timedelta(minutes=1))
        self.assertEqual(p.trips.distance_km.tolist(), [7])
        self.assertEqual(p.trips.arrival.tolist(), [m.start+pd.Timedelta(minutes=50)])
        self.assertTrue(p.available([m.start+pd.Timedelta(minutes=29)])[0])
        self.assertEqual(p.metadata["ignored_distance_km"], 5)

    def test_persistence_away_spans_gap_without_inventing_departure(self):
        m = schedule([(5, 10, 2), (40, 50, 7)])
        p = persistent_prefix(m, m.start, pd.Timedelta(hours=1), pd.Timedelta(minutes=30), False, pd.Timedelta(minutes=1))
        self.assertEqual(p.trips.distance_km.tolist(), [0, 7])
        self.assertLess(p.trips.departure.iloc[0], m.start)
        self.assertTrue(p.available([m.start+pd.Timedelta(minutes=35)])[0])
        self.assertTrue(p.available([m.start+pd.Timedelta(minutes=50)])[0])

    def test_boundary_and_beyond_horizon_demand_not_lost(self):
        for home in (False, True):
            m = schedule([(-5 if home else 5, 30, 4)])
            p = persistent_prefix(m, m.start, pd.Timedelta(hours=1), pd.Timedelta(minutes=30), home, pd.Timedelta(minutes=1))
            self.assertEqual(p.metadata["ignored_distance_km"], 4)
            self.assertTrue(p.available([m.start+pd.Timedelta(minutes=59)])[0])
        m = schedule([(-5, 40, 4)])
        p = persistent_prefix(m, m.start, pd.Timedelta(hours=1), pd.Timedelta(minutes=30), True, pd.Timedelta(minutes=1))
        self.assertEqual(p.trips.departure.iloc[0], m.start+pd.Timedelta(minutes=30))
        self.assertEqual(p.trips.distance_km.sum(), 4)

    def test_matching_state_preserves_near_term_transitions_and_trip_energy(self):
        for home, trips in ((True, [(5, 10, 2), (15, 40, 7)]),
                            (False, [(-5, 10, 2), (15, 40, 7)])):
            with self.subTest(home=home):
                m = schedule(trips)
                p = persistent_prefix(m, m.start, pd.Timedelta(hours=1),
                                      pd.Timedelta(minutes=30), home, pd.Timedelta(minutes=1))
                pd.testing.assert_frame_equal(p.trips, m.trips)
                self.assertFalse(p.metadata["prefix_applied"])
                self.assertEqual(p.metadata["ignored_distance_km"], 0)
                ev = vehicle()
                _, plan = EVMPCController(ev).build_problem(m.start, .5, forecast(ev, p))
                # Initial away trip has already been debited in measured SOC.
                self.assertAlmostEqual(plan["horizon"].trip_kwh.sum(), (9 if home else 7)*.2)

    def test_mismatched_zero_prefix_anchors_only_first_plant_step(self):
        for home, trips in ((True, [(-5, 40, 7)]), (False, [(5, 40, 7)])):
            m = schedule(trips)
            p = persistent_prefix(m, m.start, pd.Timedelta(hours=1), pd.Timedelta(0),
                                  home, pd.Timedelta(minutes=1))
            self.assertTrue(p.metadata["prefix_applied"])
            self.assertEqual(bool(p.available([m.start])[0]), home)
            ev = vehicle()
            _, plan = EVMPCController(ev).build_problem(m.start, .5, forecast(ev, p))
            self.assertAlmostEqual(plan["horizon"].trip_kwh.sum(), 1.4)

    def test_config_reserves_audi_pool_before_replay(self):
        directory = ROOT/"examples/single_home_hems/config"
        doc = json.loads((directory/"unified_control.json").read_text())
        doc.setdefault("defaults", {}).setdefault("ev", {})["forecast"] = {"mode": "scenario"}
        for home in doc.get("homes", {}).values():
            for ev in home.get("evs", {}).values():
                ev["forecast"] = {"mode": "scenario"}
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            path = Path(tmp)/"control.json"
            path.write_text(json.dumps(doc))
            cfg = load_unified(directory/"unified_home.json", path, ROOT)
        for home in cfg.neighborhood.homes:
            for ev in home.evs:
                if ev.hems.controller == "mpc":
                    self.assertGreaterEqual(ev.hems.history_days, 31)

    def test_actual_audi_interior_replay_and_scenario_issuance(self):
        from examples.five_home_block.five_home_block.ev_schedule import audi_schedule
        ev = vehicle(source="audi", history_days=31, horizon_hours=12, seed=42)
        config = replace(CONFIG, duration=pd.Timedelta(hours=1))
        path = ROOT/"examples/five_home_block/inputs/ev/audi_forecasting_15min_20vehicle.parquet"
        mobility = audi_schedule(path, ev, config)
        now = pd.Timestamp(config.start_time)
        control = scenario_control()
        control["horizon_hours"] = 12
        f = replace(forecast(ev, schedule()), ev_schedules={ev.vehicle_id: mobility})
        paths = ScenarioEVForecast(ev.vehicle_id, control).at(now, f, pd.Timedelta(hours=12))
        self.assertEqual(len(paths.paths), 5)
        self.assertLessEqual(mobility.start, now-pd.Timedelta(days=31))
        self.assertTrue(all(pd.Timestamp(m["source_end"]) <= now for m in paths.diagnostics["candidates"]))

    def test_invalid_preferences_rejected(self):
        for settings in (dict(scenario_count=0), dict(prefix_control_intervals=5),
                         dict(scenario_minutes=7), dict(knn_history_hours=0),
                         dict(lookback_days=float("nan")), dict(scenario_method="unknown")):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                scenario_control(**settings)


class EVScenarioOptimizerTests(unittest.TestCase):
    def test_scenario_solve_caps_backend_at_five_minutes(self):
        # The whole ensemble shares one solve, not 300 seconds per scenario.
        for solver, kind, key in (("SCIPY", "milp", "scipy_options"),
                                  ("CLARABEL", "convex", "time_limit"),
                                  ("GUROBI", "milp", "TimeLimit")):
            with self.subTest(solver=solver), patch.object(cp, "installed_solvers", return_value=[solver]):
                c = EVMPCController(vehicle(solver=solver, formulation=kind))
                paths = ensemble(c.ev)
                f = replace(paths.forecasts[0], providers={c.equipment_name: SimpleNamespace(at=lambda *args: paths)})
                with patch.object(cp.Problem, "solve", side_effect=cp.error.SolverError("budget test")) as call:
                    c._solve(paths.issued_at, .5, f)
                limit = call.call_args.kwargs[key]
                if solver == "SCIPY":
                    limit = limit["time_limit"]
                self.assertGreater(limit, 0)
                self.assertLessEqual(limit, 300)
                self.assertEqual(call.call_count, 1)
                self.assertTrue(c.fallback)

    def test_expired_scenario_rainflow_budget_starts_no_backend(self):
        from examples.single_home_hems.single_home_hems.controllers.ev_scenario import solve_rainflow
        c = EVMPCController(vehicle(enable_degradation=True, degradation_method="rainflow"))
        paths = ensemble(c.ev)
        problem, plan = c.build_problem(paths.issued_at, .5, paths)
        with patch.object(cp.Problem, "solve") as call:
            result, accepted = solve_rainflow(c, problem, plan, {}, deadline=0)
        call.assert_not_called()
        self.assertFalse(accepted)
        self.assertTrue(result.hems_diagnostic["timed_out"])

    def test_one_scenario_equivalence_and_first_command_sharing(self):
        for kind, solver in (("convex", "CLARABEL"), ("milp", "SCIPY")):
            ev = vehicle(formulation=kind, solver=solver, comfort_weight=1000,
                         enable_degradation=True, degradation_method="linear_throughput")
            c = EVMPCController(ev)
            paths = ensemble(ev, trips=(((30, 45, 2),),))
            single, _ = c.build_problem(paths.issued_at, .5, paths.forecasts[0])
            joint, plan = c.build_problem(paths.issued_at, .5, paths)
            solve_ocp(single, solver)
            solve_ocp(joint, solver)
            self.assertAlmostEqual(single.value, joint.value, places=5)
            paths = ensemble(ev)
            joint, plan = c.build_problem(paths.issued_at, .5, paths)
            solve_ocp(joint, solver)
            self.assertTrue(usable_solution(joint))
            a, b = plan["scenario_plans"]
            self.assertAlmostEqual(a["chemical"].value[0], b["chemical"].value[0], places=6)

    def test_aggregation_separate_max_comfort_and_trip_mean_other_terms(self):
        c = EVMPCController(vehicle(comfort_weight=1000, health_soc_weight=100,
                                   persistence_weight=1, enforce_terminal_soc=True))
        paths = ensemble(c.ev, trips=(((15, 25, 2), (45, 55, 2)), ((30, 40, 3),)))
        _, plan = c.build_problem(paths.issued_at, .5, paths)
        for i, p in enumerate(plan["scenario_plans"]):
            p["state"].value = np.full(len(p["boundaries"]), .1+i*.2)
            p["chemical"].value = np.ones(len(p["horizon"].dt))*(i+1)
            if p["trip_shortfall_variable"] is not None:
                p["trip_shortfall_variable"].value = np.full(len(p["trip_departure_blocks"]), i*.1)
        for key, total in plan["objective_terms"].items():
            values = [p["objective_terms"][key].value for p in plan["scenario_plans"]]
            self.assertAlmostEqual(total.value, max(values) if key in {"comfort", "trip_shortfall"} else np.mean(values))

    def test_rainflow_joint_solve_certifies_each_path(self):
        for kind, solver in (("convex", "CLARABEL"), ("milp", "SCIPY")):
            with self.subTest(kind=kind):
                c = EVMPCController(vehicle(formulation=kind, solver=solver,
                    enable_degradation=True, degradation_method="rainflow", health_weight=1))
                paths = ensemble(c.ev)
                f = replace(paths.forecasts[0], providers={c.equipment_name: SimpleNamespace(at=lambda *args: paths)})
                observed = list(c.soc_history)
                c._solve(paths.issued_at, .5, f)
                self.assertFalse(c.fallback, c.solve_history[-1])
                info = c.solve_history[-1]
                self.assertTrue(info["rainflow_model_certified"])
                self.assertTrue(np.all(np.asarray(info["rainflow_scenario_errors_usd"]) <= info["rainflow_scenario_tolerances_usd"]))
                self.assertEqual(len(c.plans[-1]["scenarios"]), 2)
                self.assertEqual(c.soc_history, observed)

    def test_optional_constraints_and_measured_capacity_per_path(self):
        for kind, solver in (("convex", "CLARABEL"), ("milp", "SCIPY")):
            c = EVMPCController(vehicle(formulation=kind, solver=solver,
                enforce_cycle_limit=True, max_charge_starts_24h=1, max_discharge_starts_24h=1,
                enforce_dwell_constraints=True, dwell_on_minutes=15, dwell_off_minutes=15,
                persistence_weight=.1, enforce_discharge_budget=True, discharge_budget_soc=.1,
                enforce_terminal_soc=True))
            c.current_soh, c.current_capacity = .85, 8.5
            paths = ensemble(c.ev)
            p, plan = c.build_problem(paths.issued_at, .5, paths)
            solve_ocp(p, solver)
            self.assertTrue(usable_solution(p))
            for q in plan["scenario_plans"]:
                self.assertEqual(q["capacity_kwh"], 8.5)
                self.assertLessEqual(q["discharge_future"].value, .1+1e-6)
                np.testing.assert_allclose(np.diff(q["state"].value)*8.5,
                    q["horizon"].dt*q["chemical"].value-q["horizon"].trip_kwh, atol=1e-6)

    def test_unreachable_scenario_is_retained_with_soft_trip_slack(self):
        c = EVMPCController(vehicle())
        paths = ensemble(c.ev, trips=((), ((15, 30, 1000),)))
        f = replace(paths.forecasts[0], providers={c.equipment_name: SimpleNamespace(at=lambda *args: paths)})
        c._solve(paths.issued_at, .5, f)
        self.assertFalse(c.fallback)
        self.assertGreater(c.solve_history[-1]["predicted_trip_shortfall_kwh_max"], 0)
        self.assertEqual(c.solve_history[-1]["predicted_scenarios_with_trip_shortfall"], 1)
        self.assertEqual(len(c.plans[-1]["scenarios"]), 2)
        self.assertEqual(c.next_solve, paths.issued_at+pd.Timedelta(minutes=15))

    def test_uncertified_rainflow_uses_baseline_fallback(self):
        c = EVMPCController(vehicle(enable_degradation=True, degradation_method="rainflow",
                                    rainflow_max_iterations=1))
        paths = ensemble(c.ev)
        f = replace(paths.forecasts[0], providers={c.equipment_name: SimpleNamespace(at=lambda *args: paths)})
        c._solve(paths.issued_at, .5, f)
        self.assertTrue(c.fallback)
        self.assertFalse(c.solve_history[-1]["rainflow_converged"])
        self.assertFalse(c.plans)

    def test_current_minimum_terminal_value_preserved(self):
        c = EVMPCController(vehicle(enforce_terminal_soc=True))
        paths = ensemble(c.ev)
        tariff = lambda t: .3 if 15 <= t.hour < 18 else .15
        paths = replace(paths, forecasts=tuple(replace(f, tariff_price_at=tariff) for f in paths.forecasts))
        _, plan = c.build_problem(paths.issued_at, .5, paths)
        self.assertEqual(plan["terminal_value_usd_per_kwh"], .15)
        for p in plan["scenario_plans"]:
            self.assertEqual(p["terminal_value_usd_per_kwh"], .15)

    def test_shared_command_across_first_interval_mobility_splits(self):
        for kind, solver in (("convex", "CLARABEL"), ("milp", "SCIPY")):
            c = EVMPCController(vehicle(formulation=kind, solver=solver))
            paths = ensemble(c.ev, trips=(((7, 12, 1),), ((9, 14, 1),)))
            p, plan = c.build_problem(paths.issued_at, .5, paths)
            solve_ocp(p, solver)
            self.assertTrue(usable_solution(p))
            first_powers = []
            for q in plan["scenario_plans"]:
                active = (q["horizon"].group == 0) & (q["horizon"].z == 1)
                first_powers.extend(q["chemical"].value[active])
            np.testing.assert_allclose(first_powers, first_powers[0], atol=1e-6)

    @unittest.skipUnless("GUROBI" in cp.installed_solvers(), "Gurobi not installed")
    def test_gurobi_both_formulations_all_aging_options(self):
        for kind in ("convex", "milp"):
            for method in ("none", "linear_throughput", "rainflow"):
                with self.subTest(kind=kind, method=method):
                    c = EVMPCController(vehicle(formulation=kind, solver="GUROBI",
                        enable_degradation=True, degradation_method=method))
                    paths = ensemble(c.ev)
                    f = replace(paths.forecasts[0], providers={c.equipment_name: SimpleNamespace(at=lambda *args: paths)})
                    c._solve(paths.issued_at, .5, f)
                    self.assertFalse(c.fallback, c.solve_history[-1])


class EVScenarioPipelineTests(unittest.TestCase):
    def test_short_three_case_run_fixed_clock_and_outputs(self):
        self.check_pipeline(True)

    def test_optional_twin_uses_extended_baseline_and_two_case_outputs(self):
        self.check_pipeline(False)

    def check_pipeline(self, run_twin):
        ev = vehicle(formulation="milp", solver="SCIPY", horizon_hours=1,
                     enable_degradation=True, degradation_method="linear_throughput")
        home = replace(CONFIG.homes[0], evs=(ev,))
        doc = {"defaults": {"ev": dict(controller="mpc", formulation="milp", solver="SCIPY",
            horizon_hours=1, forecast=dict(mode="scenario", scenario_count=2))}}
        doc["run_non_v2h_twin"] = run_twin
        controls = {"evs": {ev.vehicle_id: resolve_control(doc, home.home_id, "ev", ev.vehicle_id)},
                    **{d: resolve_control(doc, home.home_id, d) for d in ("hpwh", "hvac")}}
        mobility = historical()
        current_trip = schedule([(7, 12, .5)]).trips
        mobility = replace(mobility, trips=pd.concat([mobility.trips, current_trip]).sort_values("departure"))
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp, redirect_stdout(io.StringIO()):
            cfg = replace(CONFIG, homes=(home,), duration=pd.Timedelta(minutes=16), output_directory=Path(tmp))
            u = UnifiedConfiguration(cfg, {home.home_id: controls}, {home.home_id: {
                "backed_loads": ["heat_pump", "water_heater"], "water_heater_hp_only": True}}, {}, doc,
                warmup_hours=0, cooldown_hours=0)
            result = run_unified(u, mobility_by_home={home.home_id: {ev.vehicle_id: mobility}})[home.home_id]
            expected = CASES if run_twin else ("all_baseline", "advanced")
            self.assertEqual(tuple(result.cases), expected)
            saved = json.loads((Path(tmp)/"resolved_configuration.json").read_text())
            self.assertEqual(saved["cases"], list(expected))
            self.assertEqual(saved["backed_load_reference_case"], u.backed_load_reference_case)
            if not run_twin:
                self.assertFalse((Path(tmp)/home.home_id/"non_v2h_twin").exists())
                self.assertFalse((Path(tmp)/"data/non_v2h_twin_aggregate.csv").exists())
                baseline_dir = Path(tmp)/home.home_id/"all_baseline/data"
                raw_all = pd.read_csv(baseline_dir/"raw_results_all_phases.csv", index_col=0, parse_dates=True)
                self.assertEqual(raw_all.index[-1]+cfg.time_step,
                                 pd.Timestamp(cfg.start_time)+cfg.duration+pd.Timedelta(hours=1))
                self.assertTrue(pd.read_csv(baseline_dir/"evaluation_mask.csv").phase.eq("forecast_tail").any())
                loads = pd.read_csv(baseline_dir/"backed_loads.csv")
                np.testing.assert_allclose(loads.actual_backed_load_kw, loads.reference_backed_load_kw)
            name = f"EV_{ev.vehicle_id}"
            self.assertTrue(result.metrics.energy_adjusted_net_savings_usd.notna().all())
            self.assertTrue((Path(tmp)/home.home_id/"figures"/f"{name}_forecast_accuracy.png").exists())
            self.assertTrue((Path(tmp)/home.home_id/"figures/whole_home_savings_comparison.png").exists())
            for case in expected:
                row = result.solver_stats.loc[result.solver_stats.case.eq(case) & result.solver_stats.device.eq(name)].iloc[0]
                self.assertEqual(row.attempts, 0 if case == "all_baseline" else 2)
                self.assertEqual(row.failures, 0)
                raw = result.cases[case]["raw"]
                departure = mobility.start+pd.Timedelta(days=40, minutes=7)
                arrival = departure+pd.Timedelta(minutes=5)
                self.assertGreater(raw.loc[departure, f"{name} Trip Energy (kWh)"], 0)
                self.assertEqual(raw.loc[arrival, f"{name} Trip Energy (kWh)"], 0)
                self.assertTrue(raw.loc[~raw[f"{name} Parked"].astype(bool), f"{name} Electric Power (kW)"].eq(0).all())
                battery = result.metrics.loc[result.metrics.case.eq(case) & result.metrics.device.eq(name)].iloc[0]
                total = result.metrics.loc[result.metrics.case.eq(case) & result.metrics.device.eq("whole_home")].iloc[0]
                self.assertAlmostEqual(battery.initial_stored_energy_kwh, raw[f"{name} Start Stored Energy (kWh)"].iloc[0])
                self.assertAlmostEqual(battery.final_stored_energy_kwh, raw[f"{name} End Stored Energy (kWh)"].iloc[-1])
                self.assertAlmostEqual(total.energy_inventory_adjustment_usd, battery.energy_inventory_adjustment_usd)
                self.assertAlmostEqual(battery.initial_energy_value_usd_per_kwh, .15)
                self.assertTrue(raw[f"{name} End SOC (-)"].between(0, 1).all())
                self.assertTrue((-raw[f"{name} Electric Power (kW)"] <= raw[f"{name} Actual Backed Load (kW)"]+1e-7).all())
                if case != "all_baseline":
                    directory = Path(tmp)/home.home_id/case/"data"
                    plans = json.loads((directory/f"{name}_plans.json").read_text())
                    self.assertEqual(len(plans[0]["scenarios"]), 2)
                    issuance = pd.read_csv(directory/f"{name}_forecast_issuance.csv")
                    self.assertTrue(issuance.backed_load_source.eq(f"shared {u.backed_load_reference_case} trajectory").all())
                    self.assertTrue(issuance.scenario_count.eq(2).all())
                    self.assertFalse(issuance.no_scenarios_selected.any())
                    errors = pd.read_csv(directory/f"{name}_forecast_errors.csv")
                    self.assertEqual(len(errors), 2)
                    self.assertFalse(errors.full_horizon.any())
                    self.assertTrue(errors.horizon_energy_mae_kwh.isna().all())
                    self.assertTrue((directory/f"{name}_forecast_errors_by_lead.csv").exists())
                    solves = pd.read_csv(directory/f"{name}_solve_history.csv")
                    self.assertTrue(solves.scenario_count.eq(2).all())


if __name__ == "__main__":
    unittest.main()
