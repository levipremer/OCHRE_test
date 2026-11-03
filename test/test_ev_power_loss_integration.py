"""Integration checks for the opt-in power-loss EV plant and realized SOH."""
from dataclasses import replace
import datetime as dt
from pathlib import Path
import unittest
import pandas as pd

from examples.five_home_block.five_home_block.ev_equipment import BidirectionalEV
from examples.five_home_block.five_home_block.ev_schedule import EVSchedule, build_ev_schedules
from examples.single_home_hems.battery_model_comparison.nmc_models import (
    NMCProfile,
    PowerLossPlant,
)
from examples.single_home_hems.single_home_hems.controllers.ev import EVMPCController, EVHorizon
from examples.single_home_hems.single_home_hems.control_forecasts import EVScenarioSet
from examples.single_home_hems.single_home_hems.ev_scenarios import persistent_prefix
from test.test_ev_v2h import EVPlantTests, forecast, vehicle
from examples.single_home_hems.single_home_hems.control_config import load_unified


class EVPowerLossIntegrationTests(unittest.TestCase):
    def selected_ev(self):
        root = Path(__file__).resolve().parents[1]
        directory = root / "examples/single_home_hems/config"
        unified = load_unified(directory / "unified_home.json", directory / "unified_control.json", root)
        return next(ev for home in unified.neighborhood.homes if home.home_id == "home_01"
                    for ev in home.evs if ev.vehicle_id == "01A")

    def test_selected_home_configuration_opts_in(self):
        ev = self.selected_ev()
        self.assertEqual(ev.hems.battery_model, "power_loss")
        self.assertEqual(ev.hems.standby_vehicle_kw, 0.015)
        self.assertEqual(ev.hems.charge_efficiency, 0.93)
        self.assertEqual(ev.hems.discharge_efficiency, 0.93)
        self.assertEqual(ev.capacity_kwh * ev.hems.initial_soh, 51.0)

    def test_selected_week_keeps_the_audi_vehicle(self):
        root = Path(__file__).resolve().parents[1]
        directory = root / "examples/single_home_hems/config"
        unified = load_unified(directory / "unified_home.json", directory / "unified_control.json", root)
        replay = replace(unified.neighborhood, start_time=dt.datetime(2025, 12, 1),
                         duration=dt.timedelta(days=7))
        home = next(home for home in replay.homes if home.home_id == "home_01")
        self.assertEqual(build_ev_schedules(replay, home)["01A"].metadata["source_vehicle"],
                         "CA_d7b156076b3a51c8")

    def test_selected_audi_arrivals_use_signed_soc_and_allow_away_gain(self):
        # Isolate the supplied trip impulses from home-idle battery drain.
        configured = self.selected_ev()
        ev = replace(configured, hems=replace(configured.hems, standby_vehicle_kw=0.0))
        start = pd.Timestamp("2018-01-01")
        mobility = EVSchedule(pd.DataFrame([
            dict(departure=start + pd.Timedelta(minutes=1),
                 arrival=start + pd.Timedelta(minutes=2), distance_km=100,
                 arrival_net_soc_depletion_pp=5.0),
            dict(departure=start + pd.Timedelta(minutes=3),
                 arrival=start + pd.Timedelta(minutes=4), distance_km=100,
                 arrival_net_soc_depletion_pp=-2.0),
        ]), start, start + pd.Timedelta(hours=2), {"source": "audi"})
        plant = BidirectionalEV(ev_config=ev, mobility=mobility,
                                start_time=dt.datetime(2018, 1, 1),
                                duration=dt.timedelta(hours=1),
                                time_res=dt.timedelta(minutes=1),
                                verbosity=7, save_results=False)
        step = EVPlantTests().step
        rows = [step(plant, 0) for _ in range(5)]
        name = plant.results_name
        self.assertAlmostEqual(rows[2][f"{name} End SOC (-)"], ev.initial_soc_fraction - .05)
        self.assertAlmostEqual(rows[4][f"{name} End SOC (-)"], ev.initial_soc_fraction - .03)
        self.assertAlmostEqual(rows[2][f"{name} Trip Energy (kWh)"], 51 * .05)
        self.assertLess(rows[4][f"{name} Trip Energy (kWh)"], 0)
        self.assertAlmostEqual(rows[4][f"{name} Trip Energy (kWh)"], -rows[4][f"{name} Capacity (kWh)"] * .02)
        self.assertAlmostEqual(rows[4][f"{name} Trip Unserved Energy (kWh)"], 0)

    def test_selected_audi_empty_schedule_stays_home(self):
        ev = self.selected_ev()
        start = pd.Timestamp("2018-01-01")
        mobility = EVSchedule(pd.DataFrame(columns=["departure", "arrival", "distance_km"]),
                              start, start + pd.Timedelta(hours=2), {"source": "audi"})
        self.assertTrue(mobility.demand(ev.hems.consumption_kwh_per_km,
                        soc_depletion_capacity_kwh=51).empty)
        plant = BidirectionalEV(ev_config=ev, mobility=mobility,
                                start_time=dt.datetime(2018, 1, 1),
                                duration=dt.timedelta(hours=1),
                                time_res=dt.timedelta(minutes=1),
                                verbosity=7, save_results=False)
        row = EVPlantTests().step(plant, 0)
        name = plant.results_name
        self.assertTrue(row[f"{name} Parked"])
        self.assertAlmostEqual(row[f"{name} Vehicle Standby Loss (kWh)"], 0.015 / 60)
        self.assertLess(plant.soc, ev.initial_soc_fraction)

    def test_power_loss_horizon_uses_supplied_soc_instead_of_distance(self):
        ev, _ = self.plant()
        start = pd.Timestamp("2018-01-01")
        mobility = EVSchedule(pd.DataFrame([
            dict(departure=start + pd.Timedelta(minutes=15),
                 arrival=start + pd.Timedelta(minutes=30), distance_km=100,
                 arrival_net_soc_depletion_pp=5.0),
            dict(departure=start + pd.Timedelta(minutes=45),
                 arrival=start + pd.Timedelta(minutes=50), distance_km=100,
                 arrival_net_soc_depletion_pp=-2.0),
        ]), start, start + pd.Timedelta(hours=2), {"source": "audi"})
        horizon = EVHorizon.from_forecast(EVMPCController(ev), start, forecast(ev, mobility))
        self.assertAlmostEqual(horizon.trip_kwh.sum(), 51 * .03)
        self.assertAlmostEqual(horizon.trip_kwh[horizon.t[1:] == start + pd.Timedelta(minutes=30)][0], 51 * .05)
        self.assertAlmostEqual(horizon.trip_kwh[horizon.t[1:] == start + pd.Timedelta(minutes=50)][0], -51 * .02)
        self.assertNotAlmostEqual(horizon.trip_kwh.sum(), 200 * ev.hems.consumption_kwh_per_km)

    def test_scenario_soc_depletion_is_at_departure_even_if_return_is_later(self):
        ev, _ = self.plant()
        start = pd.Timestamp("2018-01-01")
        mobility = EVSchedule(pd.DataFrame([
            dict(departure=start + pd.Timedelta(minutes=15),
                 arrival=start + pd.Timedelta(minutes=50), distance_km=100,
                 arrival_net_soc_depletion_pp=5.0),
            dict(departure=start + pd.Timedelta(minutes=55),
                 arrival=start + pd.Timedelta(minutes=80), distance_km=100,
                 arrival_net_soc_depletion_pp=-2.0),
        ]), start, start + pd.Timedelta(hours=2), {"source": "audi"})
        issued = forecast(ev, mobility)
        scenarios = EVScenarioSet(start, (mobility,), (issued.backed_load_kw,),
                                  (issued,), ("sample",))
        _, plan = EVMPCController(ev).build_problem(start, .5, scenarios)
        h = plan["scenario_plans"][0]["horizon"]
        self.assertEqual(h.trip_timing, "departure")
        self.assertAlmostEqual(h.trip_kwh[h.t.get_loc(start + pd.Timedelta(minutes=15))], 51 * .05)
        self.assertAlmostEqual(h.trip_kwh[h.t.get_loc(start + pd.Timedelta(minutes=55))], -51 * .02)
        self.assertEqual(h.trip_kwh[h.t.get_loc(start + pd.Timedelta(minutes=50)) - 1], 0)
        self.assertAlmostEqual(h.trip_kwh.sum(), 51 * .03)

    def test_scenario_prefix_carries_signed_soc_depletion_to_retained_arrival(self):
        ev, _ = self.plant()
        start = pd.Timestamp("2018-01-01")
        for suppressed_pp, retained_pp in ((5.0, -2.0), (-5.0, 2.0)):
            with self.subTest(suppressed_pp=suppressed_pp, retained_pp=retained_pp):
                mobility = EVSchedule(pd.DataFrame([
                    dict(departure=start + pd.Timedelta(minutes=5),
                         arrival=start + pd.Timedelta(minutes=10), distance_km=100,
                         arrival_net_soc_depletion_pp=suppressed_pp),
                    dict(departure=start + pd.Timedelta(minutes=40),
                         arrival=start + pd.Timedelta(minutes=50), distance_km=1,
                         arrival_net_soc_depletion_pp=retained_pp),
                ]), start, start + pd.Timedelta(hours=2), {"source": "audi"})
                path = persistent_prefix(mobility, start, pd.Timedelta(hours=1),
                                         pd.Timedelta(minutes=30), True,
                                         pd.Timedelta(minutes=1))
                expected_pp = suppressed_pp + retained_pp
                self.assertEqual(len(path.trips), 1)
                self.assertAlmostEqual(path.trips.arrival_net_soc_depletion_pp.iloc[0], expected_pp)
                horizon = EVHorizon.from_forecast(EVMPCController(ev), start, forecast(ev, path))
                self.assertAlmostEqual(horizon.trip_kwh.sum(), 51 * expected_pp / 100)
                scenario_horizon = EVHorizon.from_forecast(
                    EVMPCController(ev), start, forecast(ev, path), trip_timing="departure")
                departure = scenario_horizon.t.get_loc(start + pd.Timedelta(minutes=40))
                self.assertAlmostEqual(scenario_horizon.trip_kwh[departure], 51 * expected_pp / 100)

    def plant(self, *, initial_soc=0.5, standby_vehicle_kw=0.0, **options):
        ev = replace(
            vehicle(
                battery_model="power_loss",
                standby_vehicle_kw=standby_vehicle_kw,
                charge_efficiency=0.93,
                discharge_efficiency=0.93,
                max_discharge_kw=9.6,
                initial_soh=0.85,
                **options,
            ),
            capacity_kwh=60.0,
            charger_power_kw=11.5,
            initial_soc_fraction=initial_soc,
        )
        return ev, EVPlantTests().plant(ev)

    def test_first_home_charge_matches_reviewed_51_kwh_model(self):
        ev, plant = self.plant()
        reference = PowerLossPlant(
            NMCProfile.from_csv(),
            usable_kwh=ev.capacity_kwh * ev.hems.initial_soh,
            initial_energy_soc=ev.initial_soc_fraction,
            eta_charge=0.93,
            eta_discharge=0.93,
            charger_charge_kw=ev.charger_power_kw,
            charger_discharge_kw=ev.hems.max_discharge_kw,
            max_internal_step_s=60.0,
        ).step(11.5, 60.0)

        row = EVPlantTests().step(plant, 11.5)
        name = plant.results_name
        self.assertAlmostEqual(row[f"{name} Electric Power (kW)"], reference.actual_ac_average_kw, places=9)
        self.assertAlmostEqual(row[f"{name} End SOC (-)"], reference.energy_soc, places=9)
        self.assertAlmostEqual(row[f"{name} Capacity (kWh)"], 51.0, places=9)
        self.assertEqual(row[f"{name} Parked"], True)

    def test_realized_soh_and_energy_ledger_include_battery_loss(self):
        ev, plant = self.plant(reference_cycle_count=30)
        name = plant.results_name
        previous_soh = ev.hems.initial_soh
        for request, backed in ((11.5, 9.6), (-9.6, 9.6), (0.0, 9.6)):
            row = EVPlantTests().step(plant, request, load=backed)
            ac = row[f"{name} Electric Power (kW)"]
            terminal_kwh = (ac * 0.93 if ac >= 0 else ac / 0.93) / 60
            battery_loss_kwh = (
                row[f"{name} Start Stored Energy (kWh)"] + terminal_kwh
                - row[f"{name} Trip Energy (kWh)"]
                - row[f"{name} Fade Energy Loss (kWh)"]
                - row[f"{name} End Stored Energy (kWh)"]
            )
            self.assertGreaterEqual(battery_loss_kwh, -1e-10)
            if request:
                self.assertGreater(battery_loss_kwh, 0)
            else:
                self.assertAlmostEqual(battery_loss_kwh, 0, places=10)
                self.assertEqual(row[f"{name} Reference Degradation Cost ($)"], 0)
            self.assertLessEqual(row[f"{name} End SOH (-)"], previous_soh)
            self.assertAlmostEqual(
                row[f"{name} End Capacity (kWh)"],
                ev.capacity_kwh * row[f"{name} End SOH (-)"],
                places=10,
            )
            previous_soh = row[f"{name} End SOH (-)"]
        self.assertLess(plant.soh, ev.hems.initial_soh)

    def assert_energy_ledger(self, row, name):
        self.assertAlmostEqual(
            row[f"{name} Start Stored Energy (kWh)"]
            + row[f"{name} Terminal Energy (kWh)"]
            - row[f"{name} Battery Loss (kWh)"]
            - row[f"{name} Vehicle Standby Loss (kWh)"]
            - row[f"{name} Trip Energy (kWh)"]
            - row[f"{name} Fade Energy Loss (kWh)"],
            row[f"{name} End Stored Energy (kWh)"],
            places=9,
        )

    def test_home_idle_standby_drains_pack_and_updates_realized_soh(self):
        ev, plant = self.plant(standby_vehicle_kw=0.015, reference_cycle_count=30)
        row = EVPlantTests().step(plant, 0)
        name = plant.results_name
        self.assertTrue(row[f"{name} Parked"])
        self.assertEqual(row[f"{name} Electric Power (kW)"], 0)
        self.assertEqual(row[f"{name} Terminal Energy (kWh)"], 0)
        self.assertEqual(row[f"{name} Battery Loss (kWh)"], 0)
        self.assertAlmostEqual(row[f"{name} Vehicle Standby Loss (kWh)"], 0.015 / 60, places=12)
        self.assertAlmostEqual(row[f"{name} Battery Throughput (kWh)"], 0.015 / 60, places=9)
        self.assertLess(row[f"{name} End SOH (-)"], ev.hems.initial_soh)
        self.assertAlmostEqual(
            row[f"{name} End Capacity (kWh)"],
            ev.capacity_kwh * row[f"{name} End SOH (-)"],
        )
        self.assert_energy_ledger(row, name)

    def test_standby_is_zero_during_active_charge_and_discharge(self):
        _, plant = self.plant(standby_vehicle_kw=0.015)
        name = plant.results_name
        fixture = EVPlantTests()
        for request in (11.5, -9.6):
            with self.subTest(request=request):
                row = fixture.step(plant, request, load=9.6)
                self.assertNotEqual(row[f"{name} Electric Power (kW)"], 0)
                self.assertEqual(row[f"{name} Vehicle Standby Loss (kWh)"], 0)
                self.assertGreater(row[f"{name} Battery Loss (kWh)"], 0)
                self.assert_energy_ledger(row, name)

    def test_home_command_curtailed_to_zero_is_idle(self):
        for initial_soc, request, backed_load in ((1.0, 11.5, 9.6), (0.5, -9.6, 0.0)):
            with self.subTest(initial_soc=initial_soc, request=request):
                _, plant = self.plant(initial_soc=initial_soc, standby_vehicle_kw=0.015)
                row = EVPlantTests().step(plant, request, load=backed_load)
                name = plant.results_name
                self.assertEqual(row[f"{name} Electric Power (kW)"], 0)
                self.assertAlmostEqual(row[f"{name} Vehicle Standby Loss (kWh)"], 0.015 / 60)
                self.assert_energy_ledger(row, name)

    def test_standby_is_zero_away_and_resumes_at_arrival(self):
        ev = self.selected_ev()
        start = pd.Timestamp("2018-01-01")
        mobility = EVSchedule(pd.DataFrame([
            dict(departure=start + pd.Timedelta(minutes=1),
                 arrival=start + pd.Timedelta(minutes=3), distance_km=10,
                 arrival_net_soc_depletion_pp=5.0),
        ]), start, start + pd.Timedelta(hours=2), {"source": "audi"})
        plant = BidirectionalEV(ev_config=ev, mobility=mobility,
                                start_time=dt.datetime(2018, 1, 1),
                                duration=dt.timedelta(hours=1),
                                time_res=dt.timedelta(minutes=1),
                                verbosity=7, save_results=False)
        rows = [EVPlantTests().step(plant, 0) for _ in range(4)]
        name = plant.results_name
        self.assertEqual([row[f"{name} Parked"] for row in rows], [True, False, False, True])
        self.assertEqual(
            [round(row[f"{name} Vehicle Standby Loss (kWh)"], 9) for row in rows],
            [round(0.015 / 60, 9), 0, 0, round(0.015 / 60, 9)],
        )
        self.assertEqual(rows[1][f"{name} Terminal Energy (kWh)"], 0)
        self.assertEqual(rows[2][f"{name} Terminal Energy (kWh)"], 0)
        self.assertGreater(rows[3][f"{name} Trip Energy (kWh)"], 0)
        for row in rows:
            self.assert_energy_ledger(row, name)

    def test_idle_standby_saturates_at_empty_pack(self):
        _, plant = self.plant(initial_soc=0.0, standby_vehicle_kw=0.015)
        row = EVPlantTests().step(plant, 0)
        name = plant.results_name
        self.assertEqual(row[f"{name} End SOC (-)"], 0)
        self.assertEqual(row[f"{name} End Stored Energy (kWh)"], 0)
        self.assertEqual(row[f"{name} Vehicle Standby Loss (kWh)"], 0)
        self.assert_energy_ledger(row, name)

    def test_standby_checkpoint_restore_and_reset(self):
        ev, plant = self.plant(standby_vehicle_kw=0.015)
        fixture = EVPlantTests()
        fixture.step(plant, 0)
        checkpoint = plant.battery_checkpoint()
        clone = fixture.plant(ev)
        clone.restore_battery_checkpoint(checkpoint)
        self.assertEqual(clone.battery_checkpoint(), checkpoint)
        next_row = fixture.step(plant, 0)
        cloned_row = fixture.step(clone, 0)
        name = plant.results_name
        for key in ("End SOC (-)", "End SOH (-)", "Vehicle Standby Loss (kWh)"):
            self.assertAlmostEqual(next_row[f"{name} {key}"], cloned_row[f"{name} {key}"])
        with self.assertRaises(ValueError):
            fixture.plant(replace(ev, hems=replace(ev.hems, standby_vehicle_kw=0.0)))\
                .restore_battery_checkpoint(checkpoint)
        clone.reset_time()
        self.assertAlmostEqual(clone.soc, ev.initial_soc_fraction)
        self.assertAlmostEqual(clone.soh, ev.hems.initial_soh)

    def test_repeated_model_evaluation_does_not_commit_energy_or_age(self):
        _, plant = self.plant(reference_cycle_count=30)
        plant.prepare_boundary(plant.current_time)
        before = plant.battery_checkpoint()
        plant.update_external_control({"P Setpoint": 11.5, "Backed Load (kW)": 9.6})
        plant.calculate_power_and_heat()
        first = (plant.electric_kw, plant.next_soc, plant.next_ledger.snapshot())
        plant.calculate_power_and_heat()
        second = (plant.electric_kw, plant.next_soc, plant.next_ledger.snapshot())
        self.assertEqual(second, first)
        self.assertEqual(plant.battery_checkpoint(), before)

    def test_v2h_and_energy_boundaries_remain_enforced(self):
        _, plant = self.plant()
        name = plant.results_name
        discharge = EVPlantTests().step(plant, -100, load=2)
        self.assertAlmostEqual(discharge[f"{name} Electric Power (kW)"], -2, places=9)
        self.assertEqual(discharge[f"{name} Safety Override"], 1)

        _, full = self.plant(initial_soc=0.9999)
        charge = EVPlantTests().step(full, 100)
        self.assertGreaterEqual(charge[f"{name} Electric Power (kW)"], 0)
        self.assertLess(charge[f"{name} Electric Power (kW)"], 11.5)
        self.assertAlmostEqual(charge[f"{name} End SOC (-)"], 1.0, places=9)

        _, empty = self.plant(initial_soc=0.0001)
        discharge = EVPlantTests().step(empty, -100, load=9.6)
        self.assertLess(discharge[f"{name} Electric Power (kW)"], 0)
        self.assertGreater(discharge[f"{name} Electric Power (kW)"], -9.6)
        self.assertAlmostEqual(discharge[f"{name} End SOC (-)"], 0.0, places=9)

    def test_checkpoint_reset_and_legacy_mode(self):
        ev, plant = self.plant(reference_cycle_count=30)
        fixture = EVPlantTests()
        fixture.step(plant, 11.5)
        fixture.step(plant, -9.6, load=9.6)
        checkpoint = plant.battery_checkpoint()
        clone = fixture.plant(ev)
        clone.restore_battery_checkpoint(checkpoint)
        self.assertEqual(clone.battery_checkpoint(), checkpoint)
        self.assertAlmostEqual(clone.soh, plant.soh)
        with self.assertRaises(ValueError):
            fixture.plant(replace(ev, hems=replace(ev.hems, battery_model="fixed_efficiency")))\
                .restore_battery_checkpoint(checkpoint)
        clone.reset_time()
        self.assertAlmostEqual(clone.soh, ev.hems.initial_soh)
        self.assertAlmostEqual(clone.soc, ev.initial_soc_fraction)

        legacy = fixture.plant(vehicle())
        fixture.step(legacy, 4)
        self.assertAlmostEqual(legacy.soc, 0.5 + 4 * 0.9 / 60 / 10, places=12)


if __name__ == "__main__":
    unittest.main()
