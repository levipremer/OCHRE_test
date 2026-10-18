"""Dynamic health: calibration, transactional plant aging and MPC feedback."""
from dataclasses import replace
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from test import test_ev_v2h as fixtures
from test.test_ev_v2h import vehicle, schedule, forecast, CONFIG, ROOT
from examples.five_home_block.five_home_block.ev_degradation import BatteryDegradation, RainflowLedger
from examples.single_home_hems.single_home_hems.controllers.ev import EVMPCController
from examples.single_home_hems.single_home_hems.controllers.solver import solve_ocp
from examples.single_home_hems.single_home_hems.control_config import load_unified
from examples.single_home_hems.single_home_hems.ev_metrics import ev_metrics


class DynamicHealthTests(unittest.TestCase):
    def test_calibration_and_economic_budget_have_distinct_units(self):
        ev = vehicle(initial_soh=.85)
        b = BatteryDegradation(ev.capacity_kwh, ev.hems)
        d = b.cycle_damage(.8, .6)
        self.assertAlmostEqual(d*3000, 1)
        self.assertAlmostEqual(b.calibration_fade*d*3000, .2)
        self.assertAlmostEqual(b.allowed_damage, 1.5)
        self.assertAlmostEqual(d*4500*b.damage_price, b.depreciable_usd)
        self.assertAlmostEqual((b.allowed_damage-.15/b.calibration_fade)/d, 2250)
        faded = BatteryDegradation(ev.capacity_kwh, ev.hems, current_capacity_kwh=8.5)
        self.assertEqual(b.damage_price, faded.damage_price)
        self.assertAlmostEqual(faded.throughput_usd_per_kwh*.85, b.throughput_usd_per_kwh)

    def test_streaming_matches_batch_every_prefix_and_restores_residue(self):
        ev = vehicle()
        b = BatteryDegradation(ev.capacity_kwh, ev.hems)
        rng = np.random.default_rng(12)
        for path in ([.2, .8, .4, .7, .1, .1, .6, .8], *rng.random((20, 30))):
            ledger = RainflowLedger(b, path[0])
            for j, soc in enumerate(path[1:], 1):
                before = ledger.snapshot()
                trial = ledger.preview(soc, 8.5)
                self.assertEqual(before, ledger.snapshot())
                expected = b.costs(path[:j+1])
                self.assertAlmostEqual(trial.totals[0]*b.damage_price, expected["reference_degradation_cost_usd"], places=11)
                self.assertAlmostEqual(trial.totals[1]*b.damage_price, expected["rainflow_degradation_cost_usd"], places=11)
                self.assertTrue(np.all(trial.totals >= ledger.totals))
                ledger = RainflowLedger.restore(b, json.loads(json.dumps(trial.snapshot())))
            self.assertAlmostEqual(ledger.throughput_kwh, 8.5*np.abs(np.diff(path)).sum())

    def test_invalid_health_and_nonmonotone_stress_rejected(self):
        for soh in (0, -1, 1.01, float("nan")):
            with self.assertRaises(ValueError):
                vehicle(initial_soh=soh)
        with self.assertRaisesRegex(ValueError, "Dynamic rainflow"):
            vehicle(soc_stress_coefficient=8)
        b = BatteryDegradation(10, vehicle().hems)
        with self.assertRaisesRegex(ValueError, "empty residue"):
            RainflowLedger.restore(b, {"residue": []})

    def test_preview_commit_conservation_arrival_and_checkpoint(self):
        f = fixtures.EVPlantTests()
        ev = vehicle(initial_soh=.85, reference_cycle_count=30, enable_degradation=False)
        plant = f.plant(ev, trips=[(2, 3, 1)])
        self.assertAlmostEqual(plant.capacity, 8.5)
        self.assertAlmostEqual(plant.damage, .75)
        for power in (4, -2, 0, 4, 0):
            plant.prepare_boundary(plant.current_time)
            boundary_damage = plant.damage
            plant.prepare_boundary(plant.current_time)
            self.assertEqual(plant.damage, boundary_damage)
            plant.update_external_control({"P Setpoint": power, "Backed Load (kW)": 3})
            plant.calculate_power_and_heat()
            trial = plant.next_ledger.snapshot()
            plant.calculate_power_and_heat()
            self.assertEqual(plant.next_ledger.snapshot(), trial)
            self.assertEqual(plant.damage, boundary_damage)
            row = f.step(plant, power)
            n = plant.results_name
            dc = plant.electric_kw*.9 if plant.electric_kw >= 0 else plant.electric_kw/.8
            self.assertAlmostEqual(row[f"{n} End Stored Energy (kWh)"],
                row[f"{n} Start Stored Energy (kWh)"] + dc/60 - row[f"{n} Trip Energy (kWh)"]
                - row[f"{n} Fade Energy Loss (kWh)"], places=11)
            self.assertAlmostEqual(row[f"{n} End SOH (-)"], plant.soh)
        self.assertLess(plant.soh, .85)  # Objective disabled does not disable physics.
        clone = f.plant(ev, trips=[(2, 3, 1)])
        clone.restore_battery_checkpoint(json.loads(json.dumps(plant.battery_checkpoint())))
        clone.current_time = plant.current_time
        for power in (-3, 2, 0):
            f.step(clone, power)
            f.step(plant, power)
            self.assertEqual(clone.battery_checkpoint(), plant.battery_checkpoint())
        clone.reset_time()
        self.assertAlmostEqual(clone.soh, .85)
        self.assertEqual(clone.ledger.totals.tolist(), [0, 0, 0])
        different = f.plant(replace(ev, hems=replace(ev.hems, reference_cycle_count=31)))
        with self.assertRaisesRegex(ValueError, "initialization"):
            different.restore_battery_checkpoint(plant.battery_checkpoint())

    def test_idle_no_bill_initial_damage_and_eol_is_flag_only(self):
        f = fixtures.EVPlantTests()
        plant = f.plant(vehicle(initial_soh=.69))
        row = f.step(plant, 0)
        n = plant.results_name
        self.assertEqual(row[f"{n} Reference Degradation Cost ($)"], 0)
        self.assertEqual(row[f"{n} Economic EOL Reached"], 1)
        f.step(plant, -1)
        self.assertLess(plant.soh, .69)

    def test_scoring_retains_warmup_residue_and_excludes_cooldown(self):
        f = fixtures.EVPlantTests()
        ev = replace(vehicle(initial_soh=.85, comfort_soc_min=.8), initial_soc_fraction=.2)
        plant = f.plant(ev)
        rows = [f.step(plant) for _ in range(10)]
        raw = pd.DataFrame(rows, index=pd.date_range(schedule().start, periods=10, freq="min"))
        # Standalone equipment result names differ from dwelling equipment names.
        name = f"EV_{ev.vehicle_id}"
        raw.columns = [c.replace(plant.results_name, name, 1) for c in raw.columns]
        scored = raw.iloc[3:7]
        row = ev_metrics(scored, replace(CONFIG.homes[0], evs=(ev,)),
                         pd.Series(1., index=scored.index), 1/60, previous=raw.iloc[2]).iloc[0]
        self.assertAlmostEqual(row.reference_degradation_cost_usd,
            scored[f"{name} Reference Degradation Cost ($)"].sum())
        self.assertAlmostEqual(row.initial_soh, raw[f"{name} End SOH (-)"].iloc[2])
        path = np.r_[ev.initial_soc_fraction, raw[f"{name} End SOC (-)"].iloc[:7]]
        b = plant.battery
        expected = b.costs(path)["reference_degradation_cost_usd"]-b.costs(path[:4])["reference_degradation_cost_usd"]
        self.assertAlmostEqual(row.reference_degradation_cost_usd, expected)
        self.assertLess(row.reference_degradation_cost_usd, raw[f"{name} Reference Degradation Cost ($)"].sum())
        self.assertAlmostEqual(row.soh_loss, b.calibration_fade*row.damage_increment)

    def test_mpc_freezes_measured_capacity_and_refreshes_next_solve(self):
        for kind, solver in (("convex", "CLARABEL"), ("milp", "SCIPY")):
            for method in ("none", "linear_throughput", "rainflow"):
                with self.subTest(kind=kind, method=method):
                    ev = vehicle(initial_soh=.85, formulation=kind, solver=solver,
                                 degradation_method=method, enable_degradation=method != "none")
                    c, m = EVMPCController(ev), schedule([(7, 12, 1)])
                    f, name = forecast(ev, m), c.equipment_name
                    status = {f"{name} SOC (-)": .5, f"{name} SOH (-)": .85}
                    c.controls_for_step(m.start, status, {}, f)
                    self.assertTrue(c.solve_history[-1]["successful"])
                    first = c.plans[-1]
                    self.assertEqual(first["capacity_kwh"], 8.5)
                    status[f"{name} SOH (-)"] = .84
                    c.controls_for_step(m.start+pd.Timedelta(minutes=15), status, {}, f)
                    self.assertAlmostEqual(c.plans[-1]["capacity_kwh"], 8.4)
                    self.assertEqual(first["capacity_kwh"], 8.5)
                    p, q = c.build_problem(m.start, .5, f)
                    solve_ocp(p, solver)
                    np.testing.assert_allclose(np.diff(q["state"].value)*8.4,
                        q["horizon"].dt*q["chemical"].value-q["horizon"].trip_kwh, atol=1e-6)

    def test_measurement_and_explicit_configuration_required(self):
        ev, m = vehicle(), schedule()
        c = EVMPCController(ev)
        with self.assertRaisesRegex(ValueError, "plant SOH"):
            c.controls_for_step(m.start, {c.equipment_name+" SOC (-)": .5}, {}, forecast(ev, m))
        directory = ROOT/"examples/single_home_hems/config"
        home = json.loads((directory/"unified_home.json").read_text())
        control = (directory/"unified_control.json").read_text()
        del home["homes"]["home_01"]["evs"]["01A"]["physical"]["initial_soh"]
        with patch("pathlib.Path.read_text", side_effect=[json.dumps(home), control]):
            with self.assertRaisesRegex(ValueError, "initial_soh"):
                load_unified("h.json", "c.json", ROOT)

    def test_historical_discharge_uses_capacity_at_each_action(self):
        c, m = EVMPCController(vehicle(initial_soh=.85)), schedule()
        c.discharge_samples = [(m.start-pd.Timedelta(hours=2), m.start-pd.Timedelta(hours=1), .85, 8.5),
                               (m.start-pd.Timedelta(hours=1), m.start, .8, 8.)]
        c.current_capacity = 7.
        self.assertAlmostEqual(c.discharge_history(m.start-pd.Timedelta(hours=2), m.start, normalized=True), .2)
        self.assertAlmostEqual(c.discharge_history(m.start-pd.Timedelta(hours=1.5), m.start, normalized=True), .15)

    def test_physical_exhaustion_raises_not_negative_capacity(self):
        f = fixtures.EVPlantTests()
        plant = f.plant(vehicle(initial_soh=.00001, reference_cycle_count=1))
        with self.assertRaisesRegex(RuntimeError, "physical capacity exhausted"):
            f.step(plant, -1)


if __name__ == "__main__":
    unittest.main()
