"""Unified config, forecast, fixed-clock and three-case output regressions.

Only short fixtures, not research comparison studies, are executed here.
"""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

from test.test_ev_v2h import ROOT, CONFIG, vehicle, schedule, forecast
from examples.single_home_hems.single_home_hems.control_config import load_unified, resolve_control, UnifiedConfiguration
from examples.single_home_hems.single_home_hems.control_forecasts import ExternalForecast, EVScenarioSet, UNITS
from examples.single_home_hems.single_home_hems.backed_loads import backed_power, validate_backed_loads
from examples.single_home_hems.single_home_hems.controllers.ev import EVMPCController, EVBaselineController, EVScenarioController
from examples.single_home_hems.single_home_hems.unified import run_home, make_controllers, CASES


class OptionalTwinConfigTests(unittest.TestCase):
    def test_load_optional_twin_default_and_strict_boolean(self):
        directory = ROOT/"examples/single_home_hems/config"
        document = json.loads((directory/"unified_control.json").read_text())
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            path = Path(tmp)/"control.json"
            for value in (True, False, "omitted", "false", 0, None):
                document.pop("run_non_v2h_twin", None)
                if value != "omitted":
                    document["run_non_v2h_twin"] = value
                path.write_text(json.dumps(document))
                if value is None or value == "false" or type(value) is int:
                    with self.assertRaisesRegex(ValueError, "run_non_v2h_twin must be boolean"):
                        load_unified(directory/"unified_home.json", path, ROOT)
                else:
                    cfg = load_unified(directory/"unified_home.json", path, ROOT)
                    self.assertEqual(cfg.run_non_v2h_twin, value is not False)

class UnifiedConfigTests(unittest.TestCase):
    def test_examples_load_and_split_physics_from_preferences(self):
        c = load_unified(ROOT/"examples/single_home_hems/config/unified_home.json",
                         ROOT/"examples/single_home_hems/config/unified_control.json", ROOT)
        ev = c.neighborhood.homes[0].evs[0]
        self.assertTrue(ev.hems.v2h)
        self.assertEqual(ev.hems.controller, "mpc")
        self.assertNotIn("solver", c.home_document["homes"]["home_01"]["evs"]["01A"]["physical"])
        self.assertNotIn("capacity_kwh", c.controls["home_01"]["evs"]["01A"])

    def test_independent_home_and_vehicle_overrides(self):
        doc = {"defaults": {"ev": {"solver": "SCIPY", "horizon_hours": 2}}, "homes": {
            "a": {"ev_defaults": {"control_minutes": 10}, "evs": {"one": {"solver": "CLARABEL"}}},
            "b": {"evs": {"one": {"solver": "GUROBI"}}}}}
        a = resolve_control(doc, "a", "ev", "one")
        self.assertEqual((a["solver"], a["control_minutes"], a["forecast_hours"]), ("CLARABEL", 10, 2))
        self.assertEqual(resolve_control(doc, "a", "ev", "two")["solver"], "SCIPY")
        self.assertEqual(resolve_control(doc, "b", "ev", "one")["solver"], "GUROBI")

    def test_scenario_ev_enabled_hvac_still_pass_through(self):
        c = resolve_control({"defaults": {"ev": {"controller": "mpc", "forecast": {"mode": "scenario"}}}}, "h", "ev", "v")
        self.assertEqual(c["forecast"]["scenario_method"], "knn_state")
        with self.assertRaisesRegex(NotImplementedError, "pass-through"):
            resolve_control({"defaults": {"hvac": {"controller": "mpc"}}}, "h", "hvac")
        self.assertIsInstance(EVScenarioController(vehicle()), EVMPCController)
        with self.assertRaises(ValueError):
            EVScenarioSet(pd.Timestamp("2018-01-01"), (), ())

    def test_unsupported_forecast_and_model_intervals_rejected(self):
        for prefs in ({"forecast": {"mode": "deterministic"}},
                      {"forecast": {"mode": "scenario"}, "horizon_hours": 6},
                      {"control_minutes": 5}, {"model_minutes": 10}):
            with self.subTest(prefs=prefs), self.assertRaises(ValueError):
                resolve_control({"defaults": {"hpwh": {"controller": "mpc", **prefs}}}, "h", "hpwh")

    def test_default_ev_is_not_tou_avoidant(self):
        self.assertEqual(EVBaselineController().controls_for_step(None, None, None, None), {})

    def test_mixed_hpwh_ev_solvers_and_hpwh_scenario_factory(self):
        ev = vehicle(formulation="milp", solver="SCIPY")
        home = replace(CONFIG.homes[0], evs=(ev,))
        for mode in ("perfect", "scenario"):
            doc = {"defaults": {"hpwh": {"controller": "mpc", "solver": "CLARABEL", "forecast": {"mode": mode}},
                                 "ev": {"controller": "mpc", "solver": "SCIPY", "formulation": "milp"}}}
            c = {"hpwh": resolve_control(doc, home.home_id, "hpwh"),
                 "evs": {ev.vehicle_id: resolve_control(doc, home.home_id, "ev", ev.vehicle_id)}}
            devices, _ = make_controllers(home, c, ROOT)
            self.assertEqual(devices[0].options.solver, "SCIPY")
            self.assertEqual(devices[1].solver, "CLARABEL")
            self.assertEqual(devices[1].forecast_type, mode)

    def test_unified_cli_delegates_without_running_experiment(self):
        from examples.single_home_hems.run_simulation import parse_args, main
        args = parse_args(["--home-config", str(ROOT/"examples/single_home_hems/config/unified_home.json"),
                           "--control-config", str(ROOT/"examples/single_home_hems/config/unified_control.json"),
                           "--homes", "home_01", "home_02"])
        with patch("examples.single_home_hems.single_home_hems.unified.run_unified") as runner:
            self.assertEqual(main(args), 0)
            self.assertEqual(runner.call_args.args[1], ["home_01", "home_02"])
        args.water_heater_solver = "SCIPY"
        with self.assertRaisesRegex(ValueError, "JSON"):
            main(args)


class UnifiedPhysicsTests(unittest.TestCase):
    def test_hp_excludes_er_and_unknown_components_raise(self):
        row = {"HVAC Heating Main Power (kW)": 2.0, "HVAC Heating Fan Power (kW)": .3,
               "HVAC Heating ER Power (kW)": 10, "HVAC Cooling Main Power (kW)": 0,
               "HVAC Cooling Fan Power (kW)": 0, "Water Heating Electric Power (kW)": .5}
        self.assertAlmostEqual(backed_power(row), 2.8)
        row["HVAC Heating ER Power (kW)"] = 20
        self.assertAlmostEqual(backed_power(row), 2.8)
        with self.assertRaises(ValueError):
            backed_power({}, ["heat_pump"])
        with self.assertRaises(ValueError):
            validate_backed_loads(["EV_01A"])
        with self.assertRaises(ValueError):
            validate_backed_loads(["heat_pump", "HVAC Heating"])
        from examples.five_home_block.five_home_block.ev_equipment import BidirectionalEV
        with redirect_stdout(io.StringIO()):
            plant = BidirectionalEV(ev_config=vehicle(), mobility=schedule(), start_time=CONFIG.start_time,
                duration=pd.Timedelta(minutes=2), time_res=pd.Timedelta(minutes=1), verbosity=7, save_results=False)
            actual = [0.5]
            plant.actual_backed_load = lambda: actual[0]
            plant.update(control_signal={"P Setpoint": -4, "Backed Load (kW)": 10})
            self.assertEqual(plant.electric_kw, -.5)
            actual[0] = .25
            plant.update(control_signal={"P Setpoint": -4, "Backed Load (kW)": 10})
            self.assertEqual(plant.electric_kw, -.25)
            self.assertEqual(plant.safety_override, 1)

    def test_mobility_never_triggers_an_ocp(self):
        ev, mobility = vehicle(), schedule([(7, 12, 1)])
        c = EVMPCController(ev)
        f = forecast(ev, mobility)
        for minute in range(31):
            c.controls_for_step(mobility.start+pd.Timedelta(minutes=minute),
                {f"EV_{ev.vehicle_id} SOC (-)": .5, f"EV_{ev.vehicle_id} SOH (-)": 1.}, {}, f)
        self.assertEqual([h["timestamp"] for h in c.solve_history],
                         [mobility.start+pd.Timedelta(minutes=m) for m in (0, 15, 30)])

    def test_event_split_does_not_create_extra_command(self):
        ev, mobility = vehicle(), schedule([(7, 12, 1)])
        c = EVMPCController(ev)
        p, plan = c.build_problem(mobility.start, .5, forecast(ev, mobility))
        p.solve(solver="CLARABEL")
        # Available blocks before departure and after return share the command.
        self.assertAlmostEqual(plan["charge"].value[0], plan["charge"].value[2], places=6)
        self.assertAlmostEqual(plan["chemical"].value[0], plan["chemical"].value[2], places=6)


class ExternalForecastTests(unittest.TestCase):
    def snapshot(self):
        t = schedule().start
        rows = [{"timestamp": str(t+pd.Timedelta(minutes=k)),
                 "electricity_price_usd_per_kwh": .2, "backed_load_kw": 1.0} for k in range(60)]
        return {"units": {**UNITS, "distance_km": "km"}, "snapshots": [
            {"issued_at": str(t), "data": rows, "trips": []},
            {"issued_at": str(t+pd.Timedelta(minutes=5)), "data": rows, "trips": []}]}

    def test_causal_snapshot_and_no_future_realized_load(self):
        doc = self.snapshot()
        ev, mobility = vehicle(horizon_hours=.25), schedule()
        with patch.object(Path, "read_text", return_value=json.dumps(doc)):
            provider = ExternalForecast("unused.json", f"EV_{ev.vehicle_id}", .25)
        result = provider.at(mobility.start, forecast(ev, mobility, load=99), pd.Timedelta(minutes=15))
        self.assertTrue(result.backed_load_kw.eq(1).all())
        self.assertEqual(provider.history[-1]["issued_at"], str(mobility.start))
        with self.assertRaisesRegex(ValueError, "No causal"):
            provider.at(mobility.start-pd.Timedelta(minutes=1), forecast(ev, mobility), pd.Timedelta(minutes=15))

    def test_coverage_and_units_fail_without_perfect_fallback(self):
        doc = self.snapshot()
        doc["snapshots"][0]["data"] = doc["snapshots"][0]["data"][:2]
        with patch.object(Path, "read_text", return_value=json.dumps(doc)):
            provider = ExternalForecast("unused", "EV_01A", .25)
        with self.assertRaisesRegex(ValueError, "coverage"):
            provider.at(schedule().start, forecast(vehicle(), schedule()), pd.Timedelta(minutes=15))
        doc["units"]["backed_load_kw"] = "W"
        with patch.object(Path, "read_text", return_value=json.dumps(doc)), self.assertRaises(ValueError):
            ExternalForecast("unused", "EV_01A", .25)

    def test_deterministic_ev_ocp_uses_external_prediction(self):
        ev, mobility = vehicle(horizon_hours=.25), schedule()
        with patch.object(Path, "read_text", return_value=json.dumps(self.snapshot())):
            provider = ExternalForecast("unused", f"EV_{ev.vehicle_id}", .25)
        f = replace(forecast(ev, mobility, load=99), providers={f"EV_{ev.vehicle_id}": provider})
        controller = EVMPCController(ev)
        controller.controls_for_step(mobility.start, {f"EV_{ev.vehicle_id} SOC (-)": .5, f"EV_{ev.vehicle_id} SOH (-)": 1.}, {}, f)
        self.assertTrue(controller.solve_history[0]["successful"])
        self.assertLessEqual(max(controller.plans[0]["discharge_kw"]), 1.000001)

    def test_hpwh_deterministic_does_not_copy_future_water(self):
        doc = self.snapshot()
        for snapshot in doc["snapshots"]:
            for row in snapshot["data"]:
                row.update({"Water Heating (L/min)": 1., "Clothes Washer (L/min)": 0.,
                            "Dishwasher (L/min)": 0., "Mains Temperature (C)": 10.})
        with patch.object(Path, "read_text", return_value=json.dumps(doc)):
            provider = ExternalForecast("unused", "Water Heating", .25)
        f = forecast(vehicle(), schedule())
        f.data["Water Heating (L/min)"] = 99.
        result = provider.at(schedule().start, f, pd.Timedelta(minutes=15))
        self.assertTrue(result.data["Water Heating (L/min)"].eq(1).all())


class UnifiedOutputFixtureTests(unittest.TestCase):
    def test_short_three_case_fixture_preserves_outputs_and_physics(self):
        ev = vehicle(horizon_hours=.5, enable_degradation=True)
        home = replace(CONFIG.homes[0], evs=(ev,))
        doc = {"defaults": {"ev": {"controller": "mpc", "horizon_hours": .5},
                            "hpwh": {"controller": "mpc", "horizon_hours": .5, "model": "rc"}}}
        controls = {"evs": {ev.vehicle_id: resolve_control(doc, home.home_id, "ev", ev.vehicle_id)},
                    "hpwh": resolve_control(doc, home.home_id, "hpwh"),
                    "hvac": resolve_control(doc, home.home_id, "hvac")}
        with tempfile.TemporaryDirectory(dir=ROOT) as temp, redirect_stdout(io.StringIO()):
            config = replace(CONFIG, homes=(home,), duration=pd.Timedelta(minutes=16), output_directory=Path(temp))
            unified = UnifiedConfiguration(config, {home.home_id: controls},
                {home.home_id: {"backed_loads": ["heat_pump", "water_heater"], "water_heater_hp_only": True}}, {}, doc, warmup_hours=0, cooldown_hours=0)
            result = run_home(unified, home, {ev.vehicle_id: schedule([(7, 12, .5)])})
            self.assertEqual(tuple(result.cases), CASES)
            for case in CASES:
                prefix = Path(temp)/home.home_id/case
                for name in ("raw_results.csv", "standardized.csv", "controls.csv", "hvac_training.csv", "backed_loads.csv"):
                    self.assertTrue((prefix/"data"/name).is_file(), name)
                for name in ("hvac.png", "water_heater.png", "whole_home.png", f"EV_{ev.vehicle_id}.png"):
                    self.assertTrue((prefix/"figures"/name).is_file(), name)
                raw = result.cases[case]["raw"]
                self.assertEqual(len(raw), 16)
                name = f"EV_{ev.vehicle_id}"
                p = raw[f"{name} Electric Power (kW)"]
                self.assertTrue((raw[f"{name} End SOC (-)"].between(0, 1)).all())
                self.assertTrue((-p <= raw[f"{name} Actual Backed Load (kW)"]+1e-7).all())
                if case != "advanced":
                    self.assertTrue(p.ge(0).all())
                energy = ev.initial_soc_fraction*ev.capacity_kwh + (
                    p.clip(lower=0).sum()*ev.hems.charge_efficiency + p.clip(upper=0).sum()/ev.hems.discharge_efficiency)/60
                energy -= raw[f"{name} Trip Energy (kWh)"].sum()
                self.assertAlmostEqual(energy, raw[f"{name} End SOC (-)"].iloc[-1]*ev.capacity_kwh, places=7)
            baseline = result.solver_stats[result.solver_stats.case.eq("all_baseline")]
            self.assertEqual(baseline.attempts.sum(), 0)
            home_rows = result.metrics.loc[result.metrics.device.eq("whole_home")]
            base_cost = home_rows.loc[home_rows.case.eq("all_baseline"), "cost_usd"].iloc[0]
            np.testing.assert_allclose(home_rows.operating_cost_savings_usd, base_cost-home_rows.cost_usd)
            base_age = home_rows.loc[home_rows.case.eq("all_baseline"), "reference_degradation_cost_usd"].iloc[0]
            np.testing.assert_allclose(home_rows.net_savings_usd,
                                       home_rows.operating_cost_savings_usd+base_age-home_rows.reference_degradation_cost_usd)
            for case in ("non_v2h_twin", "advanced"):
                scored = result.solver_stats[result.solver_stats.case.eq(case)]
                self.assertEqual(scored.failures.sum(), 0)
                self.assertEqual(scored.loc[scored.device.eq("Water Heating"), "attempts"].iloc[0], 2)
                self.assertEqual(scored.loc[scored.device.eq(f"EV_{ev.vehicle_id}"), "attempts"].iloc[0], 2)
            from examples.single_home_hems.single_home_hems.unified import write_reports
            write_reports(config, {home.home_id: result})
            from examples.single_home_hems.single_home_hems.unified import save_configuration
            save_configuration(unified, [home])
            self.assertEqual(json.loads((Path(temp)/"resolved_configuration.json").read_text())["cases"], list(CASES))
            self.assertTrue((Path(temp)/"reports/run_report.md").is_file())
            report = (Path(temp)/"reports/run_report.md").read_text()
            for field in ("cycle_count", "final_soc", "ocp_seconds_per_sim_timestep", "backend_solve_seconds", "fallbacks", "timeouts",
                          "max_solve_seconds", "max_ocp_build_and_solve_seconds", "health_low_soc_hours", "soc_saturation_steps",
                          "estimated_degradation_cost_usd", "reference_degradation_cost_usd", "net_savings_usd", "operating_cost_savings_usd",
                          "rainflow_master_solves", "rainflow_uncertified"):
                self.assertIn(field, report)
            self.assertEqual(len(list((Path(temp)/"reports").glob("*.md"))), 1)
            self.assertTrue((Path(temp)/"data/performance.csv").is_file())
            np.testing.assert_allclose(result.cases["non_v2h_twin"]["mapped"].water_heating_power_kw,
                                       result.cases["advanced"]["mapped"].water_heating_power_kw)
