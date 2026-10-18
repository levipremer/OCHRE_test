"""Approved EV math: signed power, soft targets, end/start budgets and MIP gap."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
import cvxpy as cp
import numpy as np
import pandas as pd

from test.test_ev_v2h import CONFIG, vehicle, schedule, forecast
from test import test_ev_v2h as plant_fixture
from examples.single_home_hems.single_home_hems.controllers.ev import EVMPCController, accept_ev_solution
from examples.single_home_hems.single_home_hems.controllers.ev import EVHorizon
from examples.single_home_hems.single_home_hems.controllers.solver import solve_ocp
from examples.single_home_hems.single_home_hems.ev_metrics import ev_metrics


class EVMathematicalTests(unittest.TestCase):
    def build(self, kind="convex", trips=(), soc=.5, price=1, load=3, **opts):
        ev = vehicle(formulation=kind, solver="SCIPY" if kind == "milp" else "CLARABEL", **opts)
        c, m = EVMPCController(ev), schedule(trips)
        f = forecast(ev, m, price=price, load=load)
        p, q = c.build_problem(m.start, soc, f)
        return c, m, f, p, q

    def test_signed_power_is_only_convex_power_decision_and_conserves_energy(self):
        c, m, f, p, q = self.build(trips=[(7, 12, 1)])
        solve_ocp(p, c.options.solver)
        self.assertEqual(p.status, cp.OPTIMAL)
        self.assertEqual({v.name() for v in p.variables()}, {"x_ev", "p_chemical"})
        h, chemical = q["horizon"], q["chemical"].value
        np.testing.assert_allclose(np.diff(q["state"].value)*10, h.dt*chemical-h.trip_kwh, atol=1e-7)
        np.testing.assert_allclose(q["ac"].value, np.where(chemical >= 0, chemical/.9, chemical*.8), atol=1e-7)
        self.assertTrue(np.all(q["charge"].value*q["discharge"].value == 0))
        self.assertAlmostEqual(chemical[1], 0, places=9)
        self.assertEqual(q["N"], 4)  # six event blocks still use four control intervals
        self.assertAlmostEqual(chemical[0], chemical[2], places=6)

    def test_stage_cost_scaling_and_zero_comfort_without_departures(self):
        _, _, _, _, q = self.build(price=2, health_soc_min=.6, health_soc_max=.7,
            health_soc_weight=3, health_weight=2, comfort_soc_min=.65, comfort_weight=4, persistence_weight=7,
            enforce_terminal_soc=True)
        q["chemical"].value = np.array([1., -1., 2., 0.])
        q["state"].value = np.array([.5, .8, .4, .9, .3])
        self.assertAlmostEqual(q["electricity"].value, .25/4*2*(12+1/.9-.8+2/.9))
        self.assertAlmostEqual(q["health_low"].value, .25/4*.3)
        self.assertAlmostEqual(q["health_high"].value, .25/4*.3)
        self.assertAlmostEqual(q["health_soc"].value, 2*3*.25/4*.6)
        self.assertEqual(q["comfort"].value, 0)
        self.assertAlmostEqual(q["persistence"].value, 7*.25/4*8)
        self.assertAlmostEqual(q["terminal"].value, 2*10/4*.2)

    def test_comfort_averages_exact_pre_departure_states_only(self):
        for kind in ("convex", "milp"):
            _, m, _, _, q = self.build(kind=kind, trips=[(7, 12, 0), (35, 40, 0), (60, 70, 0)],
                                      comfort_soc_min=.4, comfort_weight=1000)
            h = q["horizon"]
            self.assertEqual(h.t[h.departure_indices].tolist(),
                             [m.start+pd.Timedelta(minutes=v) for v in (7, 35)])
            x = np.zeros(len(h.t))  # Low away/arrival/terminal states incur no comfort cost.
            x[h.departure_indices] = [.2, .5]
            q["state"].value = x
            self.assertAlmostEqual(q["comfort"].value, 1000*.2/2)

    def test_departure_now_counts_but_past_departure_does_not(self):
        for trips, expected in (([(0, 20, 0)], 1), ([(-5, 20, 0)], 0)):
            _, _, _, _, q = self.build(trips=trips, comfort_soc_min=.4, comfort_weight=10)
            q["state"].value = np.full(len(q["boundaries"]), .1)
            self.assertEqual(len(q["horizon"].departure_indices), expected)
            self.assertAlmostEqual(q["comfort"].value, 3*expected)

    def test_departure_target_prepares_for_trip_without_post_trip_floor(self):
        for kind in ("convex", "milp"):
            c, _, _, p, q = self.build(kind=kind, trips=[(30, 45, 10)], soc=.3,
                                      comfort_soc_min=.4, comfort_weight=1000, v2h=False)
            solve_ocp(p, c.options.solver)
            self.assertEqual(p.status, cp.OPTIMAL)
            self.assertGreaterEqual(q["state"].value[q["horizon"].departure_indices[0]], .4-1e-6)
            self.assertLess(q["state"].value[-1], .4)
            self.assertLess(q["comfort"].value, 1e-4)

    def test_milp_persistence_prices_modes_not_load_following_power(self):
        c, m, f, _, _ = self.build(kind="milp", persistence_weight=8)
        c.previous_mode, c.previous_power = -1, -2.
        f.backed_load_kw.iloc[15:30] = 1.
        p, q = c.build_problem(m.start, .5, f)
        variables = {v.name(): v for v in p.variables()}
        variables["b_c"].value = np.array([0, 0, 0, 0])
        variables["b_d"].value = np.array([1, 1, 1, 1])
        self.assertEqual(q["persistence"].value, 0)
        # Discharge -> idle -> charge -> discharge = 3 one-hot changes, L1=6.
        variables["b_c"].value = np.array([0, 0, 1, 0])
        variables["b_d"].value = np.array([1, 0, 0, 1])
        self.assertAlmostEqual(q["persistence"].value, 8*.25/4*6)

    def test_negative_prices_only_rejected_for_lossy_bidirectional_intervals(self):
        with self.assertRaisesRegex(ValueError, "Negative prices.*MILP"):
            self.build(price=-1)
        for options in ({"v2h": False}, {"charge_efficiency": 1, "discharge_efficiency": 1},
                        {"load": 0}, {"kind": "milp"}, {"trips": [(0, 60, 0)]}):
            c, _, _, p, q = self.build(price=-1, **options)
            self.assertTrue(p.is_dcp())
            solve_ocp(p, c.options.solver)
            self.assertEqual(p.status, cp.OPTIMAL)

    def test_mixed_sign_prices_on_affine_and_convex_intervals(self):
        c, m, f, _, _ = self.build()
        f.data.loc[f.data.index < m.start+pd.Timedelta(minutes=15), "electricity_price_usd_per_kwh"] = -1
        f.backed_load_kw.loc[f.data.index < m.start+pd.Timedelta(minutes=15)] = 0
        p, _ = c.build_problem(m.start, .5, f)
        self.assertTrue(p.is_dcp())
        solve_ocp(p, "CLARABEL")
        self.assertEqual(p.status, cp.OPTIMAL)

    def test_target_band_soft_even_without_departure_energy(self):
        for kind in ("convex", "milp"):
            c, _, _, p, q = self.build(kind=kind, trips=[(0, 60, 5)], soc=.1,
                health_soc_min=.9, health_soc_max=.95, health_soc_weight=100)
            solve_ocp(p, c.options.solver)
            self.assertEqual(p.status, cp.OPTIMAL)
            self.assertAlmostEqual(q["state"].value[-1], 0, places=6)
            self.assertGreater(q["health_low"].value, 0)

    def test_terminal_energy_value_is_optional_and_effective(self):
        for enabled in (False, True):
            c, _, _, p, q = self.build(enforce_terminal_soc=enabled)
            solve_ocp(p, c.options.solver)
            final = q["state"].value[-1]
            if enabled:
                self.assertGreaterEqual(final, .5-1e-7)
            else:
                self.assertLess(final, .2)
        # An unavoidable trip is still feasible; terminal SOC is not a hard reserve.
        c, _, _, p, q = self.build(trips=[(0, 60, 5)], enforce_terminal_soc=True)
        solve_ocp(p, c.options.solver)
        self.assertEqual(p.status, cp.OPTIMAL)
        self.assertGreater(q["terminal"].value, 0)

    def test_cycling_window_extends_history_for_short_forecasts(self):
        for hours, future, history in ((1, 1, 23), (12, 12, 12), (18, 12, 12)):
            c = EVMPCController(vehicle(horizon_hours=hours))
            m = schedule()
            f = forecast(c.ev, m)
            idx = pd.date_range(m.start, periods=24*60, freq="min")
            f = replace(f, data=f.data.reindex(idx).ffill(), backed_load_kw=pd.Series(3., index=idx))
            h = EVHorizon.from_forecast(c, m.start, f)
            self.assertEqual((h.future_hours, h.history_hours), (future, history))
            self.assertAlmostEqual(h.budget_dt.sum(), future)

    def test_cap_counts_ends_not_starts_and_does_not_clip_active_cycle(self):
        c, m, f, _, _ = self.build(kind="milp", enforce_cycle_limit=True, max_discharge_starts_24h=1)
        c.starts[-1] = [m.start-pd.Timedelta(minutes=10)]*5
        c.ends[-1] = [m.start-pd.Timedelta(minutes=5)]
        c.previous_mode, c.previous_power = -1, -3
        p, q = c.build_problem(m.start, .8, f)
        solve_ocp(p, "SCIPY")
        self.assertEqual(p.status, cp.OPTIMAL)
        np.testing.assert_allclose(q["discharge"].value, 3)
        self.assertEqual(sum(q["starts"][-1].value), 0)
        c.previous_mode, c.previous_power = 0, 0
        p, q = c.build_problem(m.start, .8, f)
        solve_ocp(p, "SCIPY")
        np.testing.assert_allclose(q["discharge"].value, 0)

    def test_milp_dwell_holds_in_ocp_but_convex_does_not(self):
        for kind in ("milp", "convex"):
            c, m, f, _, _ = self.build(kind=kind, enforce_dwell_constraints=True, dwell_on_minutes=30)
            c.previous_mode, c.previous_power = 1, 4
            c.last_change[1] = m.start
            p, q = c.build_problem(m.start, .5, f)
            solve_ocp(p, c.options.solver)
            self.assertEqual(p.status, cp.OPTIMAL)
            if kind == "milp":
                np.testing.assert_allclose(q["charge"].value[:2], 4)
            else:
                self.assertLess(q["chemical"].value[0], 0)

    def test_discharge_budget_counts_battery_energy_and_partial_history(self):
        for kind in ("convex", "milp"):
            c, m, f, _, _ = self.build(kind=kind, enforce_discharge_budget=True, discharge_budget_soc=.2)
            c.discharge_samples = [(m.start-pd.Timedelta(hours=24), m.start-pd.Timedelta(hours=22), 2.)]
            self.assertEqual(c.discharge_history(m.start-pd.Timedelta(hours=23), m.start), 1)
            p, q = c.build_problem(m.start, .8, f)
            solve_ocp(p, c.options.solver)
            self.assertEqual(p.status, cp.OPTIMAL)
            self.assertAlmostEqual(q["discharge_past"], .1)
            self.assertLessEqual(q["discharge_future"].value, .1+1e-7)
            self.assertGreater(q["discharge_future"].value, .09)

    def test_discharge_budget_excludes_driving(self):
        c, _, _, p, q = self.build(trips=[(0, 60, 5)], enforce_discharge_budget=True, discharge_budget_soc=0)
        solve_ocp(p, c.options.solver)
        self.assertEqual(p.status, cp.OPTIMAL)
        self.assertAlmostEqual(q["state"].value[-1], .4, places=6)
        self.assertAlmostEqual(q["discharge_future"].value, 0, places=6)

    def test_infeasible_trip_fallback_tracks_target_and_preserves_solve_clock(self):
        c, m, f, _, _ = self.build(trips=[(7, 12, 100)])
        key = c.equipment_name
        action = c.controls_for_step(m.start, {f"{key} SOC (-)": .199, f"{key} SOH (-)": 1.}, {}, f)
        self.assertTrue(c.fallback)
        self.assertEqual(c.solve_history[0]["status"], "infeasible")
        self.assertAlmostEqual(action[key]["P Setpoint"], .001*10*60/.9)
        action = c.controls_for_step(m.start+pd.Timedelta(minutes=7), {f"{key} SOC (-)": .2, f"{key} SOH (-)": 1.}, {}, f)
        self.assertEqual(action[key]["P Setpoint"], 0)
        self.assertEqual(len(c.solve_history), 1)

    def test_one_interval_milp_and_convex(self):
        for kind in ("convex", "milp"):
            c, _, _, p, q = self.build(kind=kind, horizon_hours=.25)
            solve_ocp(p, c.options.solver)
            self.assertEqual(p.status, cp.OPTIMAL)
            self.assertEqual(q["N"], 1)


class EVAcceptanceAndPlantTests(unittest.TestCase):
    def test_five_percent_gap_requires_feasible_integral_incumbent(self):
        o = vehicle(formulation="milp", solver="SCIPY").hems
        v = cp.Variable(boolean=True)
        p = cp.Problem(cp.Minimize(v), [v >= 0])
        p._status, p._value = cp.USER_LIMIT, 1.
        for gap, accepted in ((.05, True), (.05001, False), (None, False), (np.nan, False), (np.inf, False)):
            for backend in ("SCIPY", "GUROBI"):
                extra = {"mip_gap": gap, "mip_dual_bound": .96} if backend == "SCIPY" else SimpleNamespace(MIPGap=gap, ObjBound=.96)
                p._solver_stats = SimpleNamespace(extra_stats=extra)
                p.hems_diagnostic = {}
                v.save_value(np.array(1.))
                self.assertEqual(accept_ev_solution(p, o), accepted)
                self.assertEqual(p.hems_diagnostic["fallback"], not accepted)
        p._solver_stats = SimpleNamespace(extra_stats={"mip_gap": .01})
        v.save_value(np.array(.5))
        self.assertFalse(accept_ev_solution(p, o))

    def test_real_scipy_gap_is_recorded(self):
        ev, m = vehicle(formulation="milp", solver="SCIPY"), schedule()
        c = EVMPCController(ev)
        c.controls_for_step(m.start, {f"{c.equipment_name} SOC (-)": .5, f"{c.equipment_name} SOH (-)": 1.}, {}, forecast(ev, m))
        self.assertTrue(c.solve_history[0]["successful"])
        self.assertLessEqual(c.solve_history[0]["relative_mip_gap"], .05)
        self.assertEqual(c.solve_history[0]["mip_gap_threshold"], .05)

    def test_plant_saturates_and_marks_runtime_soc_and_trip_deficit(self):
        fixture = plant_fixture.EVPlantTests()
        for soc, expected in ((-.1, 0), (1.1, 1)):
            plant = fixture.plant()
            plant.soc = soc
            out = fixture.step(plant, 0)
            self.assertEqual(plant.soc, expected)
            self.assertEqual(out[f"{plant.results_name} SOC Saturation"], 1)
            self.assertAlmostEqual(out[f"{plant.results_name} SOC Clipped Energy (kWh)"], 1)
        plant = fixture.plant(trips=[(0, 1, 100)])
        fixture.step(plant, 0)
        out = fixture.step(plant, 0)
        self.assertEqual(plant.soc, 0)
        self.assertEqual(out[f"{plant.results_name} Trip Unserved Energy (kWh)"], 15)
        self.assertEqual(out[f"{plant.results_name} SOC Saturation"], 1)

    def test_max_timing_and_target_metrics(self):
        ev, m = vehicle(), schedule()
        key = f"EV_{ev.vehicle_id}"
        raw = pd.DataFrame({f"{key} Electric Power (kW)": [0, 0], f"{key} SOC (-)": [.1, .9],
            f"{key} Unmet Load (kWh)": [0, 0], f"{key} Safety Override": [0, 0]},
            index=pd.date_range(m.start, periods=2, freq="min"))
        c = SimpleNamespace(equipment_name=key, action_history=[], solve_history=[dict(timestamp=t,
            successful=True, solve_seconds=s, ocp_seconds=s+1) for t, s in zip(raw.index, (1, 3))])
        row = ev_metrics(raw, replace(CONFIG.homes[0], evs=(ev,)), pd.Series(1, index=raw.index), 1/60, [c]).iloc[0]
        self.assertEqual((row.mean_solve_seconds, row.max_solve_seconds, row.max_ocp_build_and_solve_seconds), (2, 3, 4))
        self.assertAlmostEqual(row.health_low_soc_hours, .1/60)
        self.assertAlmostEqual(row.health_high_soc_hours, .1/60)


if __name__ == "__main__":
    unittest.main()
