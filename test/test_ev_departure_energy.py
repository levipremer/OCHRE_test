"""Departure impulses, comfort invariance, prefix accounting and trip-risk cost."""
from dataclasses import replace
import unittest
import cvxpy as cp
import numpy as np
import pandas as pd

from test.test_ev_v2h import vehicle, schedule, forecast, EVPlantTests
from examples.single_home_hems.single_home_hems.controllers.ev import EVMPCController
from examples.single_home_hems.single_home_hems.ev_scenarios import persistent_prefix
from examples.single_home_hems.single_home_hems.ev_forecast_metrics import score_forecasts
from examples.single_home_hems.single_home_hems.ev_metrics import ev_metrics
from examples.five_home_block.five_home_block.ev_degradation import BatteryDegradation
from test.test_ev_v2h import CONFIG


class DepartureEnergyTests(unittest.TestCase):
    def test_plant_debits_once_preserves_comfort_and_blocks_both_away_commands(self):
        f = EVPlantTests()
        plant = f.plant(vehicle(comfort_soc_min=.4), [(1, 4, 10)])
        f.step(plant, 0)
        plant.prepare_boundary(plant.current_time)
        self.assertAlmostEqual(plant.soc, .3)
        plant.prepare_boundary(plant.current_time)
        self.assertAlmostEqual(plant.soc, .3)
        first = f.step(plant, 4)
        key = plant.results_name
        self.assertAlmostEqual(first[f"{key} Pre-departure SOC (-)"], .5)
        self.assertEqual(first[f"{key} Trip Energy (kWh)"], 2)
        for result in (first, f.step(plant, -4)):
            self.assertEqual(result[f"{key} Electric Power (kW)"], 0)
            self.assertEqual(result[f"{key} Away Command Rejected"], 1)
        f.step(plant, 0)
        arrival = f.step(plant, 0)
        self.assertEqual(arrival[f"{key} Trip Energy (kWh)"], 0)
        self.assertAlmostEqual(plant.soc, .3)

    def test_initial_away_state_is_already_net_of_ongoing_trip(self):
        f = EVPlantTests()
        plant = f.plant(trips=[(-10, 1, 10)])
        f.step(plant, 0)
        f.step(plant, 0)
        self.assertEqual(plant.soc, .5)
        self.assertEqual(plant.trip_energy, 0)

    def test_measured_departure_comfort_survives_post_debit_initial_state(self):
        ev, m = vehicle(comfort_soc_min=.4, comfort_weight=1000), schedule([(0, 45, 10)])
        c = EVMPCController(ev)
        c.current_departure_soc = .5
        p, q = c.build_problem(m.start, .3, forecast(ev, m))
        p.solve(solver=ev.hems.solver)
        self.assertEqual(q["trip_count"], 0)
        self.assertEqual(q["horizon"].trip_kwh.sum(), 0)
        self.assertAlmostEqual(q["comfort"].value, 0)
        self.assertAlmostEqual(q["state"].value[0], .3)

    def test_trip_average_includes_zero_distance_trips_in_both_formulations(self):
        for kind, solver in (("convex", "CLARABEL"), ("milp", "SCIPY")):
            ev = vehicle(formulation=kind, solver=solver, initial_soh=.8)
            m = schedule([(15, 20, 10), (30, 40, 0), (45, 90, 10)])
            _, q = EVMPCController(ev).build_problem(m.start, .5, forecast(ev, m))
            q["trip_shortfall_variable"].value = np.array([1., 2.])
            self.assertEqual(q["trip_count"], 3)
            self.assertAlmostEqual(q["trip_shortfall"].value, ev.hems.trip_weight*3/(8*3))

    def test_pre_debit_capacity_bound_and_rainflow_peak(self):
        ev, m = vehicle(), schedule([(15, 30, 10)])
        p, q = EVMPCController(ev).build_problem(m.start, .95, forecast(ev, m))
        # .95 + .9/10 > 1 before departure, even though post-trip SOC < 1.
        bad = cp.Problem(p.objective, p.constraints+[q["chemical"][0] == 3.6])
        bad.solve(solver=ev.hems.solver)
        self.assertEqual(bad.status, cp.INFEASIBLE)
        p, q = EVMPCController(ev).build_problem(m.start, .5, forecast(ev, m))
        fixed = cp.Problem(p.objective, p.constraints+[q["chemical"] == 0,
                                                    q["trip_shortfall_variable"] == 0])
        fixed.solve(solver=ev.hems.solver)
        np.testing.assert_allclose(q["rainflow_state"].value[:3], [.5, .5, .3], atol=1e-7)

    def test_rainflow_certification_prices_pre_and_post_departure_states(self):
        for kind, solver in (("convex", "CLARABEL"), ("milp", "SCIPY")):
            ev = vehicle(formulation=kind, solver=solver, enable_degradation=True,
                         degradation_method="rainflow", comfort_soc_min=.55, comfort_weight=1000)
            m, c = schedule([(15, 45, 10)]), EVMPCController(ev)
            c._solve(m.start, .5, forecast(ev, m))
            self.assertFalse(c.fallback, c.solve_history[-1])
            plan = c.plans[-1]
            self.assertEqual(len(plan["rainflow_soc"]), len(plan["soc"])+1)
            b = BatteryDegradation(10, ev.hems)
            cost = b.rainflow_cost_gradient(plan["rainflow_soc"])[0]
            self.assertAlmostEqual(plan["objective_terms"]["degradation"], cost/plan["normalization_steps"], places=6)

    def test_prefix_partial_trip_retains_energy_beyond_forecast_return(self):
        m = schedule([(-5, 10, 2), (15, 90, 7)])
        p = persistent_prefix(m, m.start, pd.Timedelta(hours=1), pd.Timedelta(minutes=30), True, pd.Timedelta(minutes=1))
        self.assertEqual(p.metadata["ignored_distance_km"], 2)
        self.assertEqual(p.metadata["shifted_trip_count"], 1)
        self.assertEqual(p.trips.departure.iloc[0], m.start+pd.Timedelta(minutes=30))
        ev = vehicle()
        _, q = EVMPCController(ev).build_problem(m.start, .5, forecast(ev, p))
        self.assertAlmostEqual(q["horizon"].trip_kwh.sum(), 1.4)
        # Whole-horizon prefix shifts the crossing departure to the exclusive end.
        p = persistent_prefix(m, m.start, pd.Timedelta(hours=1), pd.Timedelta(hours=1), True, pd.Timedelta(minutes=1))
        _, q = EVMPCController(ev).build_problem(m.start, .5, forecast(ev, p))
        self.assertEqual(q["trip_count"], 0)
        self.assertEqual(p.metadata["shifted_distance_outside_horizon_km"], 7)

    def test_prefix_zero_anchors_only_first_plant_interval(self):
        m = schedule([(5, 40, 7)])
        p = persistent_prefix(m, m.start, pd.Timedelta(hours=1), pd.Timedelta(0), True, pd.Timedelta(minutes=1))
        pd.testing.assert_frame_equal(p.trips, m.trips)

    def test_departure_error_counts_trip_returning_after_horizon(self):
        m, empty = schedule([(15, 90, 10)]), schedule()
        _, _, scores = score_forecasts([(m.start, pd.Timedelta(hours=1), (empty,))],
            m, .2, 15, m.start, m.start+pd.Timedelta(hours=1))
        self.assertEqual(scores["forecast_departure_horizon_energy_mae_kwh"], 2)
        self.assertEqual(scores["forecast_next_trip_energy_mae_kwh"], 2)
        self.assertEqual(scores["forecast_arrival_energy_mae_kwh"], 0)

    def test_report_comfort_uses_pre_debit_soc(self):
        ev, m = vehicle(comfort_soc_min=.4), schedule([(0, 1, 10)])
        f = EVPlantTests()
        plant = f.plant(ev, [(0, 1, 10)])
        rows = [f.step(plant, 0), f.step(plant, 0)]
        # Equipment standalone name differs from the home factory's EV_<id>.
        raw = pd.DataFrame(rows, index=pd.date_range(m.start, periods=2, freq="min"))
        raw = raw.rename(columns=lambda s: s.replace(plant.results_name+' ', f'EV_{ev.vehicle_id} '))
        home = replace(CONFIG.homes[0], evs=(ev,))
        metrics = ev_metrics(raw, home, pd.Series(1., index=raw.index), 1/60,
                             mobility={ev.vehicle_id: m})
        self.assertEqual(metrics.iloc[0]["departure_comfort_shortfall_events"], 0)


if __name__ == "__main__":
    unittest.main()
