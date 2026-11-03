"""Scenario EV selection, persistence, two-stage OCP and solver regressions."""
from dataclasses import replace
from contextlib import redirect_stdout
import io
from types import SimpleNamespace
import json
from pathlib import Path
import tempfile
import unittest
import cvxpy as cp
import numpy as np
import pandas as pd

from test.test_ev_v2h import ROOT, CONFIG, vehicle, schedule, forecast
from examples.single_home_hems.single_home_hems.control_config import resolve_control, load_unified
from examples.single_home_hems.single_home_hems.control_config import UnifiedConfiguration
from examples.single_home_hems.single_home_hems.control_forecasts import EVScenarioSet
from examples.single_home_hems.single_home_hems.ev_scenarios import (
    ScenarioEVForecast, NoEVScenarios, persistent_prefix, FILTERS, _round_instant)
from examples.single_home_hems.single_home_hems.controllers.ev import (
    EVMPCController, _rainflow_path_cost_gradient)
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
    def test_absolute_grid_rounding_preserves_both_fall_dst_instants(self):
        zone = "America/Indiana/Indianapolis"
        grid = pd.Timedelta(minutes=15)
        for first_fold in (True, False):
            with self.subTest(first_fold=first_fold):
                stamp = pd.Timestamp("2025-11-02 01:07").tz_localize(
                    zone, ambiguous=first_fold,
                )
                self.assertEqual(_round_instant(stamp, grid, "floor"),
                                 stamp - pd.Timedelta(minutes=7))
                self.assertEqual(_round_instant(stamp, grid, "ceil"),
                                 stamp + pd.Timedelta(minutes=8))

    def issue(self, method=None, **settings):
        ev, m = vehicle(), historical()
        f = forecast(ev, schedule())
        f = replace(f, ev_schedules={ev.vehicle_id: m})
        if method is not None:
            settings["scenario_method"] = method
        provider = ScenarioEVForecast(ev.vehicle_id, scenario_control(**settings))
        return provider, provider.at(schedule().start, f, pd.Timedelta(hours=1)), f

    def test_defaults_completed_nonoverlapping_and_repeatable(self):
        provider, paths, f = self.issue()
        self.assertEqual(provider.settings["scenario_method"], "tod_dow_state_hier_limit_persistence")
        self.assertEqual(provider.settings["scenario_count"], 5)
        self.assertEqual(provider.settings["lookback_days"], 30)
        self.assertEqual(provider.settings["prefix_control_intervals"], 4)
        self.assertEqual(len(paths.paths), 5)
        np.testing.assert_allclose(paths.probabilities, .2)
        origins = sorted(pd.Timestamp(x) for x in paths.candidate_ids)
        now = schedule().start
        self.assertTrue(all(t+pd.Timedelta(hours=1) <= now for t in origins))
        self.assertTrue(all(b-a >= pd.Timedelta(hours=1) for a, b in zip(origins, origins[1:])))
        repeat = provider.at(now, f, pd.Timedelta(hours=1))
        self.assertEqual(paths.candidate_ids, repeat.candidate_ids)
        for p in paths.paths:
            self.assertTrue(p.available(pd.date_range(now, periods=60, freq="min")).all())

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
        _, paths, _ = self.issue("knn_state", lookback_days=1/24, knn_history_hours=1)
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

    def limited_persistence_fixture(self, current_home):
        ev, now = vehicle(), schedule().start
        rows = []
        for day in range(-40, 3):
            home = current_home if day == 0 else day % 2 == 0
            origin = now + pd.Timedelta(days=day)
            for departure, arrival, distance, depletion in (
                    (5 if home else -5, 10, 2, 1), (20, 25, 3, 2), (70, 80, 7, 3)):
                rows.append(dict(departure=origin+pd.Timedelta(minutes=departure),
                    arrival=origin+pd.Timedelta(minutes=arrival), distance_km=distance,
                    arrival_net_soc_depletion_pp=depletion))
        mobility = EVSchedule(pd.DataFrame(rows), now-pd.Timedelta(days=40),
                              now+pd.Timedelta(days=3), {})
        realized = replace(forecast(ev, schedule()), ev_schedules={ev.vehicle_id: mobility})
        control = scenario_control(scenario_method="tod_dow_state_hier_limit_persistence")
        control["horizon_hours"] = 2
        return ev, mobility, realized, control

    def test_limited_persistence_only_overwrites_initial_state_mismatches(self):
        now, horizon = schedule().start, pd.Timedelta(hours=2)
        for current_home in (True, False):
            with self.subTest(current_home=current_home):
                ev, mobility, realized, control = self.limited_persistence_fixture(current_home)
                limited = ScenarioEVForecast(ev.vehicle_id, control).at(now, realized, horizon)
                original_control = {**control, "forecast": {
                    **control["forecast"], "scenario_method": "tod_dow_state_hier"}}
                original = ScenarioEVForecast(ev.vehicle_id, original_control).at(now, realized, horizon)
                self.assertEqual(limited.candidate_ids, original.candidate_ids)
                np.testing.assert_array_equal(limited.probabilities, original.probabilities)
                self.assertEqual(len(limited.paths), 5)
                applied_count = 0
                for path, unconditional in zip(limited.paths, original.paths):
                    origin = pd.Timestamp(path.metadata["source_origin"])
                    source_home = bool(mobility.available([origin])[0])
                    applied = source_home != current_home
                    applied_count += applied
                    self.assertEqual(path.metadata["filter_tier"], unconditional.metadata["filter_tier"])
                    self.assertEqual(path.metadata["persistence_applied"], applied)
                    self.assertEqual(path.metadata["source_initial_state"], "home" if source_home else "away")
                    self.assertEqual(path.metadata["observed_initial_state"], "home" if current_home else "away")
                    if applied:
                        pd.testing.assert_frame_equal(path.trips, unconditional.trips)
                        self.assertEqual(path.metadata["overwritten_distance_km"], 5)
                        self.assertEqual(path.metadata["overwritten_soc_depletion_pp"], 3)
                        self.assertEqual(path.trips.distance_km.sum(), 12)
                        self.assertEqual(path.trips.arrival_net_soc_depletion_pp.sum(), 6)
                    else:
                        rows = mobility.trips.loc[(mobility.trips.arrival > origin) &
                            (mobility.trips.departure < origin+horizon)].copy()
                        for field in ("departure", "arrival"):
                            rows[field] += now-origin
                        expected = EVSchedule(rows, now, now+horizon+pd.Timedelta(minutes=1), {})
                        pd.testing.assert_frame_equal(path.trips, expected.trips)
                        # Matching home paths can depart, and matching away paths
                        # can return, during the configured persistence hour.
                        check_time = now+pd.Timedelta(minutes=6 if current_home else 12)
                        self.assertEqual(bool(path.available([check_time])[0]), not current_home)
                        self.assertEqual(bool(unconditional.available([check_time])[0]), current_home)
                self.assertGreater(applied_count, 0)
                self.assertLess(applied_count, len(limited.paths))
                self.assertEqual(limited.diagnostics["persistence_applied_count"], applied_count)

    def test_limited_persistence_is_repeatable_and_ignores_future_trip_demand(self):
        now, horizon = schedule().start, pd.Timedelta(hours=2)
        for current_home in (True, False):
            with self.subTest(current_home=current_home):
                ev, mobility, realized, control = self.limited_persistence_fixture(current_home)
                provider = ScenarioEVForecast(ev.vehicle_id, control)
                initial = provider.at(now, realized, horizon)
                changed = mobility.trips.copy()
                future = changed.arrival > now
                changed.loc[future, "distance_km"] = 10000
                changed.loc[future, "arrival_net_soc_depletion_pp"] = 50
                altered = replace(realized, ev_schedules={ev.vehicle_id: replace(mobility, trips=changed)})
                for repeated in (provider.at(now, realized, horizon), provider.at(now, altered, horizon)):
                    self.assertEqual(initial.candidate_ids, repeated.candidate_ids)
                    np.testing.assert_array_equal(initial.probabilities, repeated.probabilities)
                    self.assertEqual(initial.diagnostics["persistence_applied_count"],
                                     repeated.diagnostics["persistence_applied_count"])
                    for first, again in zip(initial.paths, repeated.paths):
                        pd.testing.assert_frame_equal(first.trips, again.trips)
                        self.assertEqual(first.metadata, again.metadata)

    def test_persistence_home_delays_and_compounds(self):
        m = schedule([(5, 10, 2), (15, 20, 3), (40, 50, 7)])
        p = persistent_prefix(m, m.start, pd.Timedelta(hours=1), pd.Timedelta(minutes=30), True, pd.Timedelta(minutes=1))
        self.assertEqual(p.trips.distance_km.tolist(), [12])
        self.assertEqual(p.trips.arrival.tolist(), [m.start+pd.Timedelta(minutes=50)])
        self.assertTrue(p.available([m.start+pd.Timedelta(minutes=29)])[0])
        self.assertEqual(p.metadata["overwritten_distance_km"], 5)

    def test_persistence_away_spans_gap_without_inventing_departure(self):
        m = schedule([(-5, 10, 2), (40, 50, 7)])
        p = persistent_prefix(m, m.start, pd.Timedelta(hours=1), pd.Timedelta(minutes=30), False, pd.Timedelta(minutes=1))
        self.assertEqual(p.trips.distance_km.tolist(), [9])
        self.assertLess(p.trips.departure.iloc[0], m.start)
        self.assertFalse(p.available([m.start+pd.Timedelta(minutes=35)])[0])
        self.assertTrue(p.available([m.start+pd.Timedelta(minutes=50)])[0])

    def test_boundary_and_beyond_horizon_demand_not_lost(self):
        for home in (False, True):
            m = schedule([(5, 30, 4)])
            p = persistent_prefix(m, m.start, pd.Timedelta(hours=1), pd.Timedelta(minutes=30), home, pd.Timedelta(minutes=1))
            self.assertEqual(p.metadata["deferred_distance_km"], 4)
            self.assertEqual(p.available([m.start+pd.Timedelta(minutes=59)])[0], home)
        m = schedule([(5, 40, 4)])
        p = persistent_prefix(m, m.start, pd.Timedelta(hours=1), pd.Timedelta(minutes=30), True, pd.Timedelta(minutes=1))
        self.assertEqual(p.trips.departure.iloc[0], m.start+pd.Timedelta(minutes=30))
        self.assertEqual(p.trips.distance_km.sum(), 4)

    def test_config_reserves_audi_pool_before_replay(self):
        directory = ROOT/"examples/single_home_hems/config"
        doc = json.loads((directory/"unified_control.json").read_text())
        doc.setdefault("defaults", {}).setdefault("ev", {})["forecast"] = {
            "mode": "scenario", "scenario_method": "knn_state"}
        for home in doc.get("homes", {}).values():
            for ev in home.get("evs", {}).values():
                ev["forecast"] = {"mode": "scenario", "scenario_method": "knn_state"}
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

    def test_actual_audi_week_maps_to_aware_oikolab_clock(self):
        from examples.five_home_block.five_home_block.ev_schedule import build_ev_schedules

        directory = ROOT/"examples/single_home_hems/config"
        unified = load_unified(directory/"unified_home.json", directory/"unified_control.json", ROOT)
        home = next(h for h in unified.neighborhood.homes if h.home_id == "home_01")
        ev = home.evs[0]
        self.assertEqual(ev.capacity_kwh, 60)
        duration = pd.Timedelta(days=7).to_pytimedelta()
        naive_start = pd.Timestamp("2025-12-01 00:00").to_pydatetime()
        aware_start = pd.Timestamp(
            "2025-12-01 00:00", tz="America/Indiana/Indianapolis"
        ).to_pydatetime()
        naive = build_ev_schedules(
            replace(unified.neighborhood, start_time=naive_start, duration=duration), home
        )[ev.vehicle_id]
        aware = build_ev_schedules(
            replace(unified.neighborhood, start_time=aware_start, duration=duration), home
        )[ev.vehicle_id]

        self.assertEqual(aware.metadata["source_vehicle"], naive.metadata["source_vehicle"])
        self.assertEqual(aware.metadata["source_start"], naive.metadata["source_start"])
        # The selected history spans the November clock change on the
        # destination calendar, so elapsed 31 days is not 31 local days.
        self.assertEqual(aware.start.tz_localize(None), naive.start)
        self.assertEqual(aware.end.tz_localize(None), naive.end)
        for column in ("departure", "arrival"):
            pd.testing.assert_series_equal(
                aware.trips[column].dt.tz_localize(None), naive.trips[column]
            )
        naive_times = pd.date_range(naive_start, periods=7 * 96, freq="15min")
        aware_times = pd.date_range(aware_start, periods=7 * 96, freq="15min")
        np.testing.assert_array_equal(aware.available(aware_times), naive.available(naive_times))

    def test_invalid_preferences_rejected(self):
        for settings in (dict(scenario_count=0), dict(prefix_control_intervals=5),
                         dict(scenario_minutes=7), dict(knn_history_hours=0),
                         dict(lookback_days=float("nan")), dict(scenario_method="unknown")):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                scenario_control(**settings)


class EVScenarioOptimizerTests(unittest.TestCase):
    def test_one_scenario_equivalence_and_first_command_sharing(self):
        for kind, solver in (("convex", "CLARABEL"), ("milp", "SCIPY")):
            ev = vehicle(formulation=kind, solver=solver, comfort_weight=1000,
                         enable_degradation=True, degradation_method="linear_throughput")
            c = EVMPCController(ev)
            paths = ensemble(ev, trips=((),))
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

    def test_scenario_trip_depletion_is_applied_at_departure(self):
        ev = vehicle(comfort_soc_min=.48, comfort_weight=1)
        controller = EVMPCController(ev)
        paths = ensemble(ev, trips=(((20, 50, 2),),))
        problem, plan = controller.build_problem(paths.issued_at, .5, paths)
        horizon = plan["scenario_plans"][0]["horizon"]
        departure = paths.issued_at + pd.Timedelta(minutes=20)
        arrival = paths.issued_at + pd.Timedelta(minutes=50)
        departure_index = horizon.t.get_loc(departure)
        arrival_index = horizon.t.get_loc(arrival)

        self.assertEqual(horizon.trip_timing, "departure")
        self.assertAlmostEqual(horizon.trip_kwh[departure_index], .4)
        self.assertAlmostEqual(horizon.trip_kwh[arrival_index - 1], 0)
        self.assertAlmostEqual(horizon.trip_kwh.sum(), .4)
        self.assertEqual(horizon.departure_indices.tolist(), [departure_index])

        scenario = plan["scenario_plans"][0]
        state = np.full(len(horizon.t), .5)
        state[departure_index] = .45
        scenario["state"].value = state
        # Low post-departure states are away, and the return at minute 50
        # starts a one-hour grace. Neither period incurs home comfort cost.
        self.assertAlmostEqual(scenario["comfort"].value, 0)
        state[departure_index] = .4
        scenario["state"].value = state
        self.assertAlmostEqual(scenario["comfort"].value, 0)
        state[0] = .4
        scenario["state"].value = state
        self.assertAlmostEqual(scenario["comfort"].value, (.48-.4)*.25/horizon.N)

        # A non-scenario forecast retains the existing arrival-time treatment.
        _, deterministic = controller.build_problem(paths.issued_at, .5, paths.forecasts[0])
        deterministic_horizon = deterministic["horizon"]
        self.assertAlmostEqual(deterministic_horizon.trip_kwh[departure_index], 0)
        self.assertAlmostEqual(deterministic_horizon.trip_kwh[arrival_index - 1], .4)

        solve_ocp(problem, ev.hems.solver)
        self.assertTrue(usable_solution(problem))
        self.assertAlmostEqual(scenario["state"].value[departure_index],
            scenario["state"].value[departure_index - 1] +
            (horizon.dt[departure_index - 1] * scenario["chemical"].value[departure_index - 1]
             - horizon.trip_kwh[departure_index]) / scenario["capacity_kwh"], places=6)

    def test_scenario_trip_depletion_is_kept_when_return_is_beyond_horizon(self):
        ev = vehicle()
        paths = ensemble(ev, trips=(((20, 80, 2),),))
        _, plan = EVMPCController(ev).build_problem(paths.issued_at, .5, paths)
        horizon = plan["scenario_plans"][0]["horizon"]
        departure = paths.issued_at + pd.Timedelta(minutes=20)

        self.assertEqual(horizon.trip_timing, "departure")
        self.assertAlmostEqual(horizon.trip_kwh[horizon.t.get_loc(departure)], .4)
        self.assertAlmostEqual(horizon.trip_kwh.sum(), .4)

    def test_scenario_away_trip_departed_before_issue_is_carried_at_issue(self):
        ev = vehicle()
        paths = ensemble(ev, trips=(((-5, 20, 2),),))
        problem, plan = EVMPCController(ev).build_problem(paths.issued_at, .5, paths)
        scenario = plan["scenario_plans"][0]
        horizon = scenario["horizon"]

        self.assertEqual(horizon.trip_timing, "departure")
        self.assertAlmostEqual(horizon.trip_kwh[0], .4)
        self.assertAlmostEqual(horizon.trip_kwh.sum(), .4)
        self.assertEqual(horizon.departure_indices.tolist(), [])
        solve_ocp(problem, ev.hems.solver)
        self.assertTrue(usable_solution(problem))
        self.assertAlmostEqual(scenario["state"].value[0], .5 - .4 / scenario["capacity_kwh"])

    def test_aggregation_max_total_comfort_and_mean_every_other_term(self):
        c = EVMPCController(vehicle(comfort_weight=1000, health_soc_weight=100,
                                   persistence_weight=1, enforce_terminal_soc=True))
        paths = ensemble(c.ev, trips=(((15, 25, 2), (45, 55, 2)), ((30, 40, 3),)))
        _, plan = c.build_problem(paths.issued_at, .5, paths)
        for i, p in enumerate(plan["scenario_plans"]):
            p["state"].value = np.full(len(p["boundaries"]), .1+i*.2)
            p["chemical"].value = np.ones(len(p["horizon"].dt))*(i+1)
            if p["trip_shortfall_variable"] is not None:
                p["trip_shortfall_variable"].value = np.full(p["trip_shortfall_variable"].shape, .01*(i+1))
        for key, total in plan["objective_terms"].items():
            values = [p["objective_terms"][key].value for p in plan["scenario_plans"]]
            self.assertAlmostEqual(total.value, max(values) if key == "comfort" else np.mean(values))

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

    def test_scenario_rainflow_includes_issue_and_future_departure_jumps(self):
        ev = vehicle(enable_degradation=True, degradation_method="rainflow", health_weight=1)
        controller = EVMPCController(ev)
        paths = ensemble(ev, trips=(((0, 15, 2), (30, 45, 3)),))
        _, plan = controller.build_problem(paths.issued_at, .5, paths)
        scenario = plan["scenario_plans"][0]
        horizon = scenario["horizon"]
        capacity = scenario["capacity_kwh"]
        np.testing.assert_allclose(horizon.trip_kwh, [.4, 0, .6, 0])

        # Synthetic coordinates avoid coincident turning points. The helper
        # must retain the measured pre-issue .5, insert the post-trip .46 at
        # issue, then insert the second pre-trip .44 before post-trip .38.
        history = np.array([.37, .58, .5])
        soc = np.array([.46, .39, .38, .41, .32])
        expanded = np.array([.37, .58, .5, .46, .39, .44, .38, .41, .32])
        battery = scenario["battery"]
        expected_cost, full_gradient = battery.rainflow_cost_gradient(expanded)
        actual_cost, gradient = _rainflow_path_cost_gradient(
            battery, history, horizon, soc, capacity)
        self.assertAlmostEqual(actual_cost, expected_cost)
        expected_gradient = np.array([full_gradient[3], full_gradient[4],
            full_gradient[5] + full_gradient[6], full_gradient[7], full_gradient[8]])
        np.testing.assert_allclose(gradient, expected_gradient, atol=1e-10)

        epsilon = 1e-6
        finite_difference = np.empty(len(soc))
        for index in range(len(soc)):
            low, high = soc.copy(), soc.copy()
            low[index] -= epsilon
            high[index] += epsilon
            low_cost, _ = _rainflow_path_cost_gradient(battery, history, horizon, low, capacity)
            high_cost, _ = _rainflow_path_cost_gradient(battery, history, horizon, high, capacity)
            finite_difference[index] = (high_cost - low_cost) / (2 * epsilon)
        np.testing.assert_allclose(gradient, finite_difference, rtol=1e-5, atol=1e-8)

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
                self.assertAlmostEqual(q["state"].value[0],
                    .5-q["horizon"].trip_kwh[0]/8.5, places=6)
                np.testing.assert_allclose(np.diff(q["state"].value)*8.5,
                    q["horizon"].dt*q["chemical"].value-
                    np.r_[q["horizon"].trip_kwh[1:], 0], atol=1e-6)

    def test_long_trip_scenario_is_retained_with_penalized_shortfall(self):
        c = EVMPCController(vehicle())
        paths = ensemble(c.ev, trips=((), ((15, 30, 1000),)))
        f = replace(paths.forecasts[0], providers={c.equipment_name: SimpleNamespace(at=lambda *args: paths)})
        c._solve(paths.issued_at, .5, f)
        self.assertFalse(c.fallback, c.solve_history[-1])
        self.assertEqual(len(c.plans[-1]["scenarios"]), 2)
        no_trip, long_trip = c.plans[-1]["scenarios"]
        self.assertAlmostEqual(no_trip["total_predicted_trip_shortfall_kwh"], 0)
        # At most ten kWh of battery capacity can serve the 200 kWh request.
        self.assertGreaterEqual(long_trip["total_predicted_trip_shortfall_kwh"], 190.-1e-6)
        self.assertGreater(long_trip["objective_terms"]["trip_shortfall"], 0)
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

    def test_approved_daily_average_terminal_value_shared_by_scenarios(self):
        c = EVMPCController(vehicle(enforce_terminal_soc=True))
        paths = ensemble(c.ev)
        def tariff(t):
            return .3 if 15 <= t.hour < 18 else .15
        paths = replace(paths, forecasts=tuple(replace(f, tariff_price_at=tariff) for f in paths.forecasts))
        _, plan = c.build_problem(paths.issued_at, .5, paths)
        self.assertAlmostEqual(plan["terminal_value_usd_per_kwh"], .16875)
        for p in plan["scenario_plans"]:
            self.assertAlmostEqual(p["terminal_value_usd_per_kwh"], .16875)

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



if __name__ == "__main__":
    unittest.main()
