"""Approved minute rounding and physical event-accounting regressions."""
import datetime as dt
from dataclasses import replace
import unittest

import pandas as pd

from examples.five_home_block.five_home_block.config import ConfigurationError
from examples.five_home_block.five_home_block.ev_schedule import EVSchedule, normalize_ev_trips
from examples.five_home_block.five_home_block.ev_equipment import BidirectionalEV
from test.test_ev_v2h import CONFIG, vehicle


class EVMinuteTimingTests(unittest.TestCase):
    start = pd.Timestamp("2018-01-01")

    def trips(self, intervals):
        return pd.DataFrame([
            {"departure": self.start + pd.Timedelta(seconds=departure),
             "arrival": self.start + pd.Timedelta(seconds=arrival),
             "distance_km": distance, "energy_kwh": energy}
            for departure, arrival, distance, energy in intervals
        ])

    def test_short_trips_round_outward_and_keep_original_timestamps(self):
        for departure, arrival in ((435, 465), (430, 440)):
            original = self.trips([(departure, arrival, 2.5, 0.5)])
            normalized = normalize_ev_trips(original)
            row = normalized.iloc[0]
            self.assertEqual(row.departure, self.start + pd.Timedelta(minutes=7))
            self.assertEqual(row.arrival, self.start + pd.Timedelta(minutes=8))
            self.assertEqual((row.distance_km, row.energy_kwh), (2.5, 0.5))
            self.assertEqual(row.original_departures, (original.departure.iloc[0].isoformat(),))
            pd.testing.assert_frame_equal(normalized, normalize_ev_trips(normalized))

    def test_rounding_overlaps_merge_demand_once(self):
        original = self.trips([(430, 440, 2, 0.4), (450, 470, 3, 0.6), (480, 510, 4, 0.8)])
        mobility = EVSchedule(original, self.start, self.start + pd.Timedelta(hours=1), {})
        self.assertEqual(len(mobility.trips), 1)
        row = mobility.trips.iloc[0]
        self.assertEqual((row.distance_km, row.energy_kwh, row.merged_trips), (9, 1.8, 3))
        self.assertEqual(row.arrival, self.start + pd.Timedelta(minutes=9))
        self.assertEqual(len(row.original_departures), 3)
        pd.testing.assert_frame_equal(mobility.trips, normalize_ev_trips(mobility.trips))

    def test_invalid_original_duration_rejected_before_rounding(self):
        for departure, arrival in ((435, 435), (445, 435)):
            with self.assertRaisesRegex(ValueError, "positive"):
                normalize_ev_trips(self.trips([(departure, arrival, 1, 0.2)]))

    def test_shifted_minute_grid_consumes_rounded_trip_exactly_once(self):
        ev = vehicle()
        mobility = EVSchedule(self.trips([(435, 465, 5, 1)]), self.start,
                              self.start + pd.Timedelta(hours=1), {})
        plant = BidirectionalEV(ev_config=ev, mobility=mobility,
            start_time=self.start + pd.Timedelta(minutes=5), duration=dt.timedelta(minutes=10),
            time_res=dt.timedelta(minutes=1), verbosity=7, save_results=False)
        served = 0
        for _ in range(5):
            plant.prepare_boundary(plant.current_time)
            plant.prepare_boundary(plant.current_time)
            row = plant.update(control_signal={"P Setpoint": 0})
            served += row[f"{plant.results_name} Trip Energy (kWh)"]
        self.assertAlmostEqual(served, 1)
        self.assertAlmostEqual(plant.soc, 0.4)

    def test_new_ev_rejects_plant_grid_that_can_skip_events(self):
        ev = vehicle()
        configured = replace(CONFIG, homes=(replace(CONFIG.homes[0], evs=(ev,)),))
        for config in (replace(configured, time_step=dt.timedelta(minutes=15)),
                       replace(configured, start_time=dt.datetime(2018, 1, 1, 0, 0, 15))):
            with self.assertRaisesRegex(ConfigurationError, "minute"):
                config.validate()
        configured.validate()
        # Opted-out legacy simulations retain their existing timestep support.
        replace(CONFIG, time_step=dt.timedelta(minutes=15)).validate()

    def test_thermal_prelude_leaves_ev_state_unchanged_until_activation(self):
        ev = replace(vehicle(), initial_soc_fraction=0.1)
        mobility = EVSchedule(self.trips([(435, 465, 5, 1)]), self.start,
                              self.start + pd.Timedelta(hours=1), {})
        plant = BidirectionalEV(ev_config=ev, mobility=mobility,
            start_time=self.start - pd.Timedelta(minutes=2), duration=dt.timedelta(minutes=5),
            time_res=dt.timedelta(minutes=1), verbosity=7, save_results=False)
        plant.activation_start = self.start
        initial = plant.battery_checkpoint()
        for _ in range(2):
            plant.update()
            self.assertEqual(plant.electric_kw, 0)
            self.assertEqual(plant.battery_checkpoint(), initial)
        plant.update()
        self.assertGreater(plant.electric_kw, 0)
        self.assertGreater(plant.soc, initial["soc"])


class EVSolverConfigurationTests(unittest.TestCase):
    def test_default_and_legacy_time_limits_have_one_effective_value(self):
        self.assertEqual(vehicle().hems.solver_time_limit_seconds, 300)
        for options in ({"TimeLimit": 60}, {"time_limit": 60}, {"scipy_options": {"time_limit": 60}}):
            configured = vehicle(solver_options=options).hems
            self.assertEqual(configured.solver_time_limit_seconds, 60)
            self.assertEqual(configured.solver_options, {})
        self.assertEqual(vehicle(solver_time_limit_seconds=12).hems.solver_time_limit_seconds, 12)

    def test_conflicting_or_invalid_time_limits_fail_early(self):
        for options in ({"solver_time_limit_seconds": 90, "solver_options": {"TimeLimit": 60}},
                        {"solver_time_limit_seconds": 0}, {"solver_time_limit_seconds": float("inf")}):
            with self.assertRaisesRegex(ValueError, "time limit"):
                vehicle(**options)


if __name__ == "__main__":
    unittest.main()
