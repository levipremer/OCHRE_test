"""Energy conservation, full trip demand, configuration and replay regressions."""
import datetime as dt
from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import cvxpy as cp

from examples.five_home_block.five_home_block.config import load_configuration
from examples.five_home_block.five_home_block.ev_config import EVOptions, validate_home_evs
from examples.five_home_block.five_home_block.ev_schedule import EVSchedule, audi_schedule, build_ev_schedules
from examples.five_home_block.five_home_block.ev_equipment import BidirectionalEV
from examples.single_home_hems.single_home_hems.controllers.ev import EVMPCController
from examples.single_home_hems.single_home_hems.forecast import Forecast

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_configuration(ROOT/"examples/five_home_block/config", ROOT)


def vehicle(**options):
    base = dict(consumption_kwh_per_km=0.2, charge_efficiency=0.9, discharge_efficiency=0.8,
                max_discharge_kw=4, health_min_soc=0.2, health_max_soc=0.8,
                controller="mpc", v2h=True, horizon_hours=1, control_minutes=15,
                health_low_weight=0, health_high_weight=0)
    base.update(options)
    return replace(CONFIG.homes[0].evs[0], capacity_kwh=10, charger_power_kw=4,
                   initial_soc_fraction=0.5, hems=EVOptions(**base))


def schedule(trips=()):
    start = pd.Timestamp("2018-01-01")
    rows = [dict(departure=start+pd.Timedelta(minutes=a), arrival=start+pd.Timedelta(minutes=b),
                 distance_km=d) for a, b, d in trips]
    return EVSchedule(pd.DataFrame(rows, columns=["departure", "arrival", "distance_km"]),
                      start, start+pd.Timedelta(days=2), {})


def forecast(ev, mobility, price=1.0, load=3.0):
    index = pd.date_range(mobility.start, periods=120, freq="min")
    return Forecast.from_schedule(pd.DataFrame(index=index), lambda t: price,
                                  ev_schedules={ev.vehicle_id: mobility},
                                  backed_load_kw=pd.Series(load, index=index))


class EVConfigurationTests(unittest.TestCase):
    def test_legacy_configs_stay_opted_out(self):
        self.assertTrue(all(ev.hems is None for home in CONFIG.homes for ev in home.evs))
        from examples.single_home_hems.run_simulation import parse_args
        self.assertEqual(parse_args([]).controller, "none")

    def test_only_one_v2h_vehicle(self):
        with self.assertRaisesRegex(ValueError, "Only one"):
            validate_home_evs([vehicle(), replace(vehicle(), vehicle_id="second")])
        validate_home_evs([vehicle(), vehicle(v2h=False)])

    def test_solver_and_parameter_validation(self):
        with self.assertRaises(ValueError):
            vehicle(formulation="milp", solver="CLARABEL")
        with self.assertRaises(ValueError):
            vehicle(charge_efficiency=0)
        with self.assertRaises(ValueError):
            vehicle(health_min_soc=0.9)
        with patch.object(cp, "installed_solvers", return_value=["SCIPY"]):
            with self.assertRaisesRegex(ValueError, "not installed"):
                EVMPCController(vehicle(solver="GUROBI"))


class EVPlantTests(unittest.TestCase):
    def plant(self, ev=None, trips=()):
        return BidirectionalEV(ev_config=ev or vehicle(), mobility=schedule(trips),
            start_time=dt.datetime(2018, 1, 1), duration=dt.timedelta(hours=2),
            time_res=dt.timedelta(minutes=1), verbosity=7, save_results=False)

    def step(self, plant, power=None, load=3):
        return plant.update(control_signal={"P Setpoint": power, "Backed Load (kW)": load} if power is not None else None)

    def test_efficiency_and_charge_discharge_limits(self):
        plant = self.plant()
        self.step(plant, 100)
        self.assertAlmostEqual(plant.soc, 0.5+4*0.9/60/10)
        old = plant.soc
        self.step(plant, -100, load=2)
        self.assertAlmostEqual(plant.soc, old-2/0.8/60/10)
        self.assertEqual(plant.electric_kw, -2)

    def test_baseline_charges_to_target_min_and_never_discharges(self):
        plant = self.plant(replace(vehicle(), initial_soc_fraction=0.199))
        self.step(plant)
        self.assertAlmostEqual(plant.soc, 0.2)
        self.step(plant)
        self.assertEqual(plant.electric_kw, 0)

    def test_trip_shortfall_is_reported_and_no_energy_is_created(self):
        plant = self.plant(trips=[(0, 2, 100)])
        first = self.step(plant, -4)
        self.assertAlmostEqual(first[f"{plant.results_name} Unmet Load (kWh)"], 15)
        self.assertEqual(plant.trip_energy, 5)
        self.assertEqual(plant.trip_unmet, 15)
        self.assertEqual(plant.electric_kw, 0)
        self.step(plant, 4)
        self.step(plant, 0)
        self.assertEqual(plant.soc, 0)
        self.assertEqual(plant.trip_energy, 0)
        self.assertEqual(plant.trip_unmet, 0)

    def test_return_does_not_reset_soc_and_discharge_requires_capability(self):
        plant = self.plant(vehicle(v2h=False), trips=[(0, 1, 5)])
        self.step(plant, 0)
        self.step(plant, -4)
        self.assertAlmostEqual(plant.soc, 0.4)
        self.assertEqual(plant.electric_kw, 0)
        plant.reset_time()
        self.assertEqual(plant.soc, 0.5)


class EVOCPTests(unittest.TestCase):
    def solve(self, ev, mobility, soc=0.5, **kwargs):
        controller = EVMPCController(ev)
        problem, plan = controller.build_problem(mobility.start, soc, forecast(ev, mobility, **kwargs))
        problem.solve(solver=ev.hems.solver)
        self.assertIn(problem.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        return controller, plan

    def test_both_formulations_allow_bounded_unreachable_trip_shortfall(self):
        for formulation, solver in (("convex", "CLARABEL"), ("milp", "SCIPY")):
            with self.subTest(formulation=formulation):
                ev, m = vehicle(formulation=formulation, solver=solver), schedule([(15, 30, 100)])
                problem, plan = EVMPCController(ev).build_problem(m.start, .5, forecast(ev, m))
                problem = cp.Problem(problem.objective, problem.constraints+[plan["chemical"] == 0])
                problem.solve(solver=solver)
                self.assertEqual(problem.status, cp.OPTIMAL)
                self.assertAlmostEqual(plan["trip_shortfall_by_block"].value.sum(), 15, places=5)
                self.assertAlmostEqual(plan["state"].value[-1], 0, places=5)

    def test_milp_modes_exclusive_and_discharge_load_capped(self):
        _, plan = self.solve(vehicle(formulation="milp", solver="SCIPY"), schedule(), load=1)
        self.assertTrue(np.all(plan["discharge"].value <= 1+1e-7))
        self.assertTrue(np.all(plan["charge"].value*plan["discharge"].value < 1e-6))
        self.assertGreater(plan["discharge"].value.sum(), 0)

    def test_departure_inside_horizon_includes_demand_even_if_return_is_outside(self):
        _, plan = self.solve(vehicle(), schedule([(30, 180, 100)]))
        self.assertEqual(plan["horizon"].trip_kwh.sum(), 20)
        self.assertIsNotNone(plan["trip_shortfall_variable"])
        self.assertGreater(plan["trip_shortfall_by_block"].value.sum(), 0)

    def test_both_formulations_consume_full_trip_energy(self):
        ev = vehicle(trip_weight=1000, health_soc_weight=1000, health_soc_min=0.8)
        for kind, solver in (("convex", "CLARABEL"), ("milp", "SCIPY")):
            e = replace(ev, hems=replace(ev.hems, formulation=kind, solver=solver))
            _, plan = self.solve(e, schedule([(15, 30, 10)]), soc=0.8, price=0)
            self.assertAlmostEqual(plan["pre_departure_state"].value[1]-plan["state"].value[1], .2, places=6)

    def test_default_shortfall_weight_preserves_feasible_trip_against_health_cost(self):
        # Consuming this trip lowers SOC below the health band. The approved
        # default must not avoid that health cost by inventing unserved driving.
        for formulation, solver in (("convex", "CLARABEL"), ("milp", "SCIPY")):
            ev = vehicle(formulation=formulation, solver=solver,
                         health_soc_weight=1000, health_soc_min=.8)
            self.assertEqual(ev.hems.trip_weight, 10000)
            _, plan = self.solve(ev, schedule([(15, 30, 10)]), soc=.8, price=0)
            self.assertAlmostEqual(plan["trip_shortfall_by_block"].value.sum(), 0, places=5)
            self.assertAlmostEqual(plan["pre_departure_state"].value[1]-plan["state"].value[1], .2, places=5)

    def test_shortfall_penalty_uses_faded_capacity_without_horizon_dilution(self):
        for hours in (1, 2):
            ev = vehicle(initial_soh=.8, horizon_hours=hours)
            m = schedule([(15, 30, 100)])
            p, plan = EVMPCController(ev).build_problem(m.start, .5, forecast(ev, m))
            cp.Problem(p.objective, p.constraints+[plan["chemical"] == 0]).solve(solver=ev.hems.solver)
            capacity = plan["capacity_kwh"]
            self.assertAlmostEqual(capacity, 8)
            self.assertAlmostEqual(plan["trip_shortfall_by_block"].value.sum(), 16, places=5)
            self.assertAlmostEqual(plan["trip_shortfall"].value, 10000*16/8, places=3)

    def test_cycle_cap_and_projection(self):
        ev = vehicle(formulation="milp", solver="SCIPY", max_discharge_starts_24h=0)
        _, plan = self.solve(ev, schedule())
        self.assertAlmostEqual(plan["discharge"].value.sum(), 0)
        controller = EVMPCController(vehicle(cycling="dwell", dwell_on_minutes=30))
        controller.previous_mode = 1
        controller.last_change[1] = schedule().start
        mode, _, _ = controller._project(schedule().start+pd.Timedelta(minutes=1), -3, 0.5, True, 3)
        self.assertEqual(mode, -1)
        mode, power, _ = controller._project(schedule().start+pd.Timedelta(minutes=1), -3, 0.5, False, 3)
        self.assertEqual((mode, power), (0, 0))

    def test_rolling_history_expires(self):
        controller = EVMPCController(vehicle())
        controller.starts[1] = [schedule().start-pd.Timedelta(hours=25), schedule().start-pd.Timedelta(hours=1)]
        controller._observe(schedule().start, 0, pd.Timedelta(minutes=1))
        self.assertEqual(len(controller.starts[1]), 1)

    def test_all_planned_variants_solve(self):
        for cycling, cap in (("none", None), ("persistence", None), ("dwell", None),
                             ("none", 3), ("persistence", 3), ("dwell", 3)):
            self.solve(vehicle(formulation="milp", solver="SCIPY", cycling=cycling,
                       persistence_weight=0.01, max_charge_starts_24h=cap, max_discharge_starts_24h=cap), schedule())

    def test_scipy_convex_and_one_interval_horizon(self):
        self.solve(vehicle(solver="SCIPY", horizon_hours=0.25), schedule())

    def test_event_boundaries_do_not_trigger_solve_and_failure_is_logged(self):
        ev = vehicle()
        controller = EVMPCController(ev)
        mobility = schedule([(7, 12, 2)])
        with patch.object(cp.Problem, "solve", side_effect=cp.error.SolverError("test failure")):
            controls = controller.controls_for_step(mobility.start,
                {f"EV_{ev.vehicle_id} SOC (-)": 0.5, f"EV_{ev.vehicle_id} SOH (-)": 1.}, {}, forecast(ev, mobility))
        self.assertEqual(controller.next_solve, mobility.start+pd.Timedelta(minutes=15))
        self.assertFalse(controller.solve_history[0]["successful"])
        self.assertEqual(controls[f"EV_{ev.vehicle_id}"]["P Setpoint"], 0)

    def test_threshold_projection_and_safety_override_reporting(self):
        controller = EVMPCController(vehicle(projection="threshold", projection_threshold=0.75))
        t = schedule().start
        self.assertEqual(controller._project(t, 2, 0.5, True, 3)[:2], (0, 0))
        self.assertEqual(controller._project(t, 3.5, 0.5, True, 3)[:2], (1, 4))
        controller = EVMPCController(vehicle(cycling="dwell"))
        controller.previous_mode = -1
        controller.last_change[-1] = t
        self.assertTrue(controller._project(t+pd.Timedelta(minutes=1), -3, 0.5, False, 3)[2])

    def test_hard_soc_bounds_with_zero_trip_penalty(self):
        for formulation, solver in (("milp", "SCIPY"), ("convex", "CLARABEL")):
            _, plan = self.solve(vehicle(formulation=formulation, solver=solver, trip_weight=0),
                                 schedule([(15, 30, 100)]), soc=.1)
            self.assertGreaterEqual(min(plan["state"].value), -1e-7)
            self.assertLessEqual(max(plan["state"].value), 1+1e-7)
            self.assertGreaterEqual(plan["trip_shortfall_by_block"].value.sum(), 18.1-1e-6)

    def test_rolling_cycle_constraint_counts_expiring_history(self):
        ev = vehicle(formulation="milp", solver="SCIPY", max_discharge_starts_24h=1)
        controller = EVMPCController(ev)
        mobility = schedule()
        controller.ends[-1] = [mobility.start-pd.Timedelta(hours=22, minutes=30)]
        problem, plan = controller.build_problem(mobility.start, 0.8, forecast(ev, mobility))
        problem.solve(solver="SCIPY")
        self.assertEqual(problem.status, cp.OPTIMAL)
        self.assertTrue(np.allclose(plan["discharge"].value, 0))
        later = mobility.start+pd.Timedelta(hours=1)
        problem, plan = controller.build_problem(later, .8, forecast(ev, mobility))
        problem.solve(solver="SCIPY")
        self.assertEqual(problem.status, cp.OPTIMAL)
        self.assertGreater(plan["discharge"].value.sum(), 0)


class EVAdapterTests(unittest.TestCase):
    def test_audi_distance_is_independent_of_soc_capacity_and_efficiency(self):
        index = pd.date_range("2019-01-01", periods=56*96, freq="15min", tz="America/Los_Angeles")
        away = (index.hour == 10)
        data = pd.DataFrame(dict(vehicle_key="test", timestamp_local=index, is_home=(~away).astype(int),
                                arrival_indicator=((index.hour == 10) & (index.minute == 45)).astype(int),
                                arrival_distance_km=np.where((index.hour == 10) & (index.minute == 45), 12, np.nan),
                                arrival_net_soc_depletion_pp=-5.0))
        ev = vehicle(source="audi", history_days=7)
        config = replace(CONFIG, duration=dt.timedelta(days=3))
        with patch("pandas.read_parquet", return_value=data):
            first = audi_schedule("dummy.parquet", ev, config)
            other = audi_schedule("dummy.parquet", replace(ev, capacity_kwh=100,
                                  hems=replace(ev.hems, consumption_kwh_per_km=0.4)), config)
        pd.testing.assert_frame_equal(first.trips, other.trips)
        self.assertTrue(first.trips.distance_km.eq(12).all())
        self.assertTrue(first.trips.arrival_net_soc_depletion_pp.eq(-5).all())
        self.assertTrue(np.allclose(first.demand(0.4).energy_kwh, 2*first.demand(0.2).energy_kwh))

    def test_real_audi_and_nhts_replay(self):
        config = replace(CONFIG, duration=dt.timedelta(hours=3))
        for source in ("audi", "nhts"):
            ev = vehicle(source=source, history_days=7, seed=42,
                source_file="examples/five_home_block/inputs/ev/audi_forecasting_15min_20vehicle.parquet" if source == "audi" else None)
            home = replace(CONFIG.homes[0], evs=(ev,))
            first = build_ev_schedules(config, home)[ev.vehicle_id]
            second = build_ev_schedules(config, home)[ev.vehicle_id]
            pd.testing.assert_frame_equal(first.trips, second.trips)
            self.assertTrue(first.trips.distance_km.ge(0).all())

    def test_audi_both_cohorts_and_file_formats(self):
        config = replace(CONFIG, duration=dt.timedelta(days=3))
        for count in (20, 100):
            ev = vehicle(source="audi", history_days=30, seed=42)
            stem = ROOT/f"examples/five_home_block/inputs/ev/audi_forecasting_15min_{count}vehicle"
            parquet = audi_schedule(str(stem)+".parquet", ev, config)
            csv = audi_schedule(str(stem)+".csv.gz", ev, config)
            pd.testing.assert_frame_equal(parquet.trips, csv.trips, check_dtype=False)
            self.assertLessEqual(parquet.start, pd.Timestamp(config.start_time)-pd.Timedelta(days=30))
            self.assertGreaterEqual(parquet.end, pd.Timestamp(config.start_time)+config.duration+pd.Timedelta(hours=12))

    def test_nhts_wrap_and_physics_independence(self):
        config = replace(CONFIG, start_time=dt.datetime(2018, 12, 31), duration=dt.timedelta(days=3))
        ev = vehicle(history_days=7, seed=42)
        home = replace(CONFIG.homes[0], evs=(ev,))
        first = build_ev_schedules(config, home)[ev.vehicle_id]
        changed = replace(ev, capacity_kwh=100, hems=replace(ev.hems, consumption_kwh_per_km=0.4))
        second = build_ev_schedules(config, replace(home, evs=(changed,)))[ev.vehicle_id]
        pd.testing.assert_frame_equal(first.trips, second.trips)
        self.assertTrue((first.trips.departure >= pd.Timestamp("2019-01-01")).any())

    def test_invalid_schedule_is_rejected(self):
        with self.assertRaises(ValueError):
            schedule([(10, 20, -1)])
        with self.assertRaises(ValueError):
            schedule([(10, 20, 2), (15, 30, 3)])


if __name__ == "__main__":
    unittest.main()
