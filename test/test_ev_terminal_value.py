"""Daily-tariff chemical-energy terminal value; no moving SOC floor."""
from dataclasses import replace
import unittest
import cvxpy as cp
import numpy as np
import pandas as pd

from test.test_ev_v2h import vehicle, schedule, forecast
from examples.single_home_hems.single_home_hems.forecast import TimeOfUsePrice
from examples.single_home_hems.single_home_hems.controllers.ev import (
    EVMPCController, daily_terminal_energy_value)
from examples.single_home_hems.single_home_hems.controllers.solver import solve_ocp


class EVTerminalValueTests(unittest.TestCase):
    def test_full_day_time_average_not_horizon_or_availability_average(self):
        ev, m = vehicle(), schedule([(0, 60, 0)])
        f = replace(forecast(ev, m), tariff_price_at=TimeOfUsePrice().price_at)
        expected = (21*.15+3*.30)/24
        self.assertAlmostEqual(daily_terminal_energy_value(f, m.start), expected)
        self.assertAlmostEqual(daily_terminal_energy_value(f, m.start+pd.Timedelta(hours=16)), expected)
        # Forecast rows cover just two hours. Future realized prices are not used.
        f.data["electricity_price_usd_per_kwh"] = 999
        self.assertAlmostEqual(daily_terminal_energy_value(f, m.start), expected)
        # A 30-minute tariff block is time-weighted, not counted as a full hour.
        f = replace(f, tariff_price_at=lambda t: .3 if t.hour == 15 and t.minute < 30 else .1)
        self.assertAlmostEqual(daily_terminal_energy_value(f, m.start), (23.5*.1+.5*.3)/24)

    def test_affine_sign_units_faded_capacity_and_normalization(self):
        for kind in ("convex", "milp"):
            for hours in (1, 2):
                with self.subTest(kind=kind, hours=hours):
                    ev = vehicle(formulation=kind, solver="SCIPY" if kind == "milp" else "CLARABEL",
                                 initial_soh=.85, horizon_hours=hours, enforce_terminal_soc=True)
                    c, m = EVMPCController(ev), schedule([(7, 12, 1)])
                    f = replace(forecast(ev, m), tariff_price_at=TimeOfUsePrice().price_at)
                    p, q = c.build_problem(m.start, .5, f)
                    self.assertTrue(p.is_dcp())
                    self.assertTrue(q["terminal"].is_affine())
                    self.assertEqual(q["N"], 4*hours)  # event splits do not change N
                    self.assertAlmostEqual(q["terminal_value_usd_per_kwh"], .16875)
                    for final in (.3, .5, .7):
                        x = np.full(len(q["boundaries"]), .5)
                        x[-1] = final
                        q["state"].value = x
                        self.assertAlmostEqual(q["terminal"].value, .16875*8.5*(.5-final)/(4*hours))

    def test_missing_or_nonfinite_tariff_fails_only_when_enabled(self):
        m = schedule()
        for enabled in (False, True):
            ev = vehicle(enforce_terminal_soc=enabled)
            c = EVMPCController(ev)
            for tariff in (None, lambda t: np.nan):
                f = replace(forecast(ev, m), tariff_price_at=tariff)
                if enabled:
                    with self.assertRaisesRegex(ValueError, "full-day tariff"):
                        c.build_problem(m.start, .5, f)
                else:
                    _, q = c.build_problem(m.start, .5, f)
                    q["state"].value = np.full(len(q["boundaries"]), .5)
                    self.assertEqual(q["terminal"].value, 0)

    def test_retired_multiplier_warns_and_cannot_scale_value(self):
        ev, m = vehicle(enforce_terminal_soc=True, terminal_soc_weight=100000), schedule()
        with self.assertWarnsRegex(UserWarning, "retired and ignored"):
            c = EVMPCController(ev)
        _, q = c.build_problem(m.start, .5, forecast(ev, m, price=.2))
        q["state"].value = np.array([.5, .5, .5, .5, .4])
        self.assertAlmostEqual(q["terminal"].value, .2*10*.1/4)

    def test_peak_discharge_can_reduce_terminal_soc(self):
        for kind in ("convex", "milp"):
            ev = vehicle(formulation=kind, solver="SCIPY" if kind == "milp" else "CLARABEL",
                         enforce_terminal_soc=True)
            m, c = schedule(), EVMPCController(ev)
            f = replace(forecast(ev, m, price=.3), tariff_price_at=TimeOfUsePrice().price_at)
            p, q = c.build_problem(m.start, .5, f)
            solve_ocp(p, ev.hems.solver)
            self.assertLess(q["state"].value[-1], .5-.1)
            self.assertGreater(q["terminal"].value, 0)

    def test_measured_capacity_refresh_and_saved_plan_accounting(self):
        for kind in ("convex", "milp"):
            ev = vehicle(formulation=kind, solver="SCIPY" if kind == "milp" else "CLARABEL",
                         enforce_terminal_soc=True)
            c, m = EVMPCController(ev), schedule()
            f = replace(forecast(ev, m, price=.3), tariff_price_at=TimeOfUsePrice().price_at)
            for i, soh in enumerate((.85, .84)):
                status = {f"{c.equipment_name} SOC (-)": .5, f"{c.equipment_name} SOH (-)": soh}
                c.controls_for_step(m.start+pd.Timedelta(minutes=15*i), status, {}, f)
                self.assertTrue(c.solve_history[-1]["successful"])
                plan = c.plans[-1]
                self.assertAlmostEqual(plan["initial_energy_kwh"], 10*soh*.5)
                self.assertAlmostEqual(plan["terminal_energy_kwh"], 10*soh*plan["soc"][-1])
                self.assertAlmostEqual(plan["objective_terms"]["terminal"],
                    plan["terminal_value_usd_per_kwh"]*(plan["initial_energy_kwh"]-plan["terminal_energy_kwh"])/4)
            self.assertEqual(c.plans[0]["capacity_kwh"], 8.5)

    def test_terminal_credit_with_all_degradation_methods_and_backends(self):
        backends = [("convex", "CLARABEL"), ("milp", "SCIPY")]
        if "GUROBI" in cp.installed_solvers():
            backends += [("convex", "GUROBI"), ("milp", "GUROBI")]
        for kind, solver in backends:
            for method in ("none", "linear_throughput", "rainflow"):
                with self.subTest(kind=kind, solver=solver, method=method):
                    ev = vehicle(formulation=kind, solver=solver, initial_soh=.85,
                                 enforce_terminal_soc=True, enable_degradation=method != "none",
                                 degradation_method=method)
                    c, m = EVMPCController(ev), schedule()
                    f = replace(forecast(ev, m, price=.15), tariff_price_at=TimeOfUsePrice().price_at)
                    status = {f"{c.equipment_name} SOC (-)": .5, f"{c.equipment_name} SOH (-)": .85}
                    c.controls_for_step(m.start, status, {}, f)
                    self.assertTrue(c.solve_history[-1]["successful"], c.solve_history[-1])
                    if method == "none":
                        self.assertLess(c.plans[-1]["objective_terms"]["terminal"], 0)


if __name__ == "__main__":
    unittest.main()
