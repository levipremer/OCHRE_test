"""Artificial return timing preserves individual source trips and demand fields."""
from dataclasses import replace
from copy import deepcopy
import datetime as dt
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from examples.five_home_block.five_home_block.ev_schedule import (
    EVSchedule, audi_schedule, delay_trip_returns,
)
from test.test_ev_v2h import CONFIG, ROOT, vehicle


class ReturnDelayTests(unittest.TestCase):
    def test_mobility_document_option_roundtrips_and_is_not_a_control_weight(self):
        from examples.single_home_hems.single_home_hems.control_config import (
            EV_CONTROL, load_unified_documents,
        )
        home_file = ROOT / "examples/single_home_hems/config/unified_home_sunday_ideal_audi.json"
        control_file = ROOT / "examples/single_home_hems/config/unified_control.json"
        homes, controls = json.loads(home_file.read_text()), json.loads(control_file.read_text())
        baseline = load_unified_documents(homes, controls, home_file, ROOT)
        self.assertEqual(baseline.neighborhood.homes[0].evs[0].hems.return_delay_minutes, 0)
        self.assertNotIn("return_delay_minutes", EV_CONTROL)
        self.assertNotIn("return_delay_collision_policy", EV_CONTROL)
        shifted_doc = deepcopy(homes)
        shifted_doc["homes"]["home_01"]["evs"]["01A"]["mobility"]["return_delay_minutes"] = 90
        shifted_doc["homes"]["home_01"]["evs"]["01A"]["mobility"]["return_delay_collision_policy"] = "cap_before_next_departure"
        shifted = load_unified_documents(shifted_doc, controls, home_file, ROOT)
        self.assertEqual(shifted.neighborhood.homes[0].evs[0].hems.return_delay_minutes, 90)
        self.assertEqual(shifted.neighborhood.homes[0].evs[0].hems.return_delay_collision_policy,
                         "cap_before_next_departure")
        again = load_unified_documents(shifted.home_document, shifted.control_document, home_file, ROOT)
        self.assertEqual(again.neighborhood.homes[0].evs[0].hems, shifted.neighborhood.homes[0].evs[0].hems)
        self.assertNotIn("return_delay_minutes", homes["homes"]["home_01"]["evs"]["01A"]["mobility"])

    def test_complete_history_and_replay_keep_departures_and_signed_demand(self):
        index = pd.date_range("2019-01-01", periods=56 * 96, freq="15min",
                              tz="America/Los_Angeles")
        away = (index.hour >= 9) & (index.hour < 17)
        arrival = (index.hour == 17) & (index.minute == 0)
        signed = np.where(index.day % 2, 6., -2.)
        data = pd.DataFrame({
            "vehicle_key": "test", "timestamp_local": index,
            "is_home": (~away).astype(int), "arrival_indicator": arrival.astype(int),
            "arrival_distance_km": np.where(arrival, 18., np.nan),
            "arrival_net_soc_depletion_pp": np.where(arrival, signed, np.nan),
            "arrival_inferred_soc_depletion_pp": np.where(arrival, 7., np.nan),
            "arrival_possible_away_charging": (arrival & (signed < 0)).astype(int),
            "episode_id": [f"episode-{i}" for i in range(len(index))],
        })
        original = data.copy(deep=True)
        ev = vehicle(source="audi", source_vehicle="test", history_days=30)
        config = replace(CONFIG, duration=dt.timedelta(days=7))
        with patch("pandas.read_parquet", return_value=data):
            baseline = audi_schedule("dummy.parquet", ev, config)
            delayed = audi_schedule("dummy.parquet", replace(
                ev, hems=replace(ev.hems, return_delay_minutes=90)), config)
        pd.testing.assert_frame_equal(data, original)
        self.assertEqual(baseline.metadata["source_start"], delayed.metadata["source_start"])
        self.assertEqual((baseline.start, baseline.end), (delayed.start, delayed.end))
        self.assertEqual(len(baseline.trips), len(delayed.trips))
        for field in ("departure", "distance_km", "arrival_net_soc_depletion_pp",
                      "arrival_inferred_soc_depletion_pp", "arrival_possible_away_charging",
                      "source_episode_ids", "original_departures", "merged_trips"):
            pd.testing.assert_series_equal(baseline.trips[field], delayed.trips[field])
        pd.testing.assert_series_equal(delayed.trips.arrival,
                                       baseline.trips.arrival + pd.Timedelta(minutes=90))
        np.testing.assert_array_equal(baseline.demand(.25, soc_depletion_capacity_kwh=51).energy_kwh,
                                      delayed.demand(.25, soc_depletion_capacity_kwh=51).energy_kwh)
        self.assertTrue(delayed.trips.arrival_net_soc_depletion_pp.lt(0).any())
        self.assertTrue(delayed.trips.departure.lt(pd.Timestamp(config.start_time)).any())
        self.assertTrue(delayed.trips.departure.ge(pd.Timestamp(config.start_time)).any())
        # These same complete shifted paths supply historical scenarios and
        # perfect mobility: no issue-time override or knowledge is introduced.
        for original_return in baseline.trips.arrival:
            if baseline.start <= original_return < baseline.end:
                self.assertTrue(baseline.available([original_return])[0])
                self.assertFalse(delayed.available([original_return])[0])
        self.assertEqual(delayed.metadata["return_delay_minutes"], 90)
        self.assertEqual(delayed.metadata["return_delay_collision_policy"], "error")

    def test_actual_ideal_history_caps_only_one_return_and_eval_delays_all_six(self):
        start = pd.Timestamp("2025-11-30", tz="America/Indiana/Indianapolis")
        config = replace(CONFIG, start_time=start.to_pydatetime(), duration=dt.timedelta(hours=176))
        ev = vehicle(source="audi", source_vehicle="CA_299a889dac408996", seed=43,
                     history_days=31, horizon_hours=8, forecast_margin_hours=8)
        path = ROOT / "examples/five_home_block/inputs/ev/audi_forecasting_15min_100vehicle.parquet"
        baseline = audi_schedule(path, ev, config)
        delayed = audi_schedule(path, replace(ev, hems=replace(
            ev.hems, return_delay_minutes=90,
            return_delay_collision_policy="cap_before_next_departure")), config)
        self.assertEqual(baseline.metadata["source_start"], "2020-02-02 00:00:00")
        self.assertEqual(baseline.metadata["source_start"], delayed.metadata["source_start"])
        self.assertEqual(len(baseline.trips), len(delayed.trips))
        for field in baseline.trips.columns.difference(["arrival", "original_arrivals"]):
            pd.testing.assert_series_equal(baseline.trips[field], delayed.trips[field])
        self.assertEqual(delayed.metadata["return_delay_capped_trip_count"], 1)
        capped = delayed.trips.loc[delayed.trips.return_delay_capped]
        self.assertEqual(len(capped), 1)
        row = capped.iloc[0]
        self.assertEqual(row.undelayed_arrival, pd.Timestamp("2025-11-19 18:30", tz=start.tz))
        self.assertEqual(row.arrival, pd.Timestamp("2025-11-19 19:29", tz=start.tz))
        self.assertEqual(row.return_delay_next_departure, pd.Timestamp("2025-11-19 19:30", tz=start.tz))
        self.assertEqual(row.actual_return_delay_minutes, 59.)
        np.testing.assert_array_equal(
            (delayed.trips.arrival - baseline.trips.arrival).dt.total_seconds() / 60,
            delayed.trips.actual_return_delay_minutes)
        evaluation = delayed.trips.loc[(delayed.trips.departure >= start) &
                                       (delayed.trips.departure < start + pd.Timedelta(hours=168))]
        self.assertEqual(len(evaluation), 6)
        self.assertTrue(evaluation.actual_return_delay_minutes.eq(90).all())
        self.assertFalse(evaluation.return_delay_capped.any())
        cap = delayed.metadata["return_delay_caps"][0]
        self.assertEqual(cap["actual_delay_minutes"], 59.)
        self.assertEqual(cap["requested_delay_minutes"], 90)
        # Source signed SOC depletion remains the input to the unchanged
        # current-capacity conversion rather than a newly frozen kWh demand.
        for capacity in (51., 50.):
            np.testing.assert_array_equal(
                baseline.demand(.25, soc_depletion_capacity_kwh=capacity).energy_kwh,
                delayed.demand(.25, soc_depletion_capacity_kwh=capacity).energy_kwh)

    def test_cap_preserves_individual_trips_and_does_not_touch_outside_coverage(self):
        rows = pd.DataFrame([
            dict(departure=pd.Timestamp("2018-01-01 09:00"),
                 arrival=pd.Timestamp("2018-01-01 18:30"), distance_km=10.,
                 arrival_net_soc_depletion_pp=-2.),
            dict(departure=pd.Timestamp("2018-01-01 19:30"),
                 arrival=pd.Timestamp("2018-01-01 20:30"), distance_km=2.,
                 arrival_net_soc_depletion_pp=3.),
            dict(departure=pd.Timestamp("2018-01-02 09:00"),
                 arrival=pd.Timestamp("2018-01-02 18:30"), distance_km=12.,
                 arrival_net_soc_depletion_pp=4.),
        ])
        shifted = delay_trip_returns(rows, 90, collision_policy="cap_before_next_departure")
        self.assertEqual(len(shifted), 3)
        self.assertEqual(shifted.actual_return_delay_minutes.tolist(), [59., 90., 90.])
        np.testing.assert_array_equal(shifted.departure, rows.departure)
        np.testing.assert_array_equal(shifted.arrival_net_soc_depletion_pp, rows.arrival_net_soc_depletion_pp)
        outside = delay_trip_returns(rows, 90, start="2018-01-02", end="2018-01-03",
                                     collision_policy="cap_before_next_departure")
        self.assertEqual(len(outside), 1)
        self.assertFalse(outside.return_delay_capped.any())

    def test_shift_precedes_window_selection_and_keeps_edge_trip_complete(self):
        rows = pd.DataFrame([
            dict(departure=pd.Timestamp("2018-01-01 20:00"),
                 arrival=pd.Timestamp("2018-01-01 23:30"), distance_km=10.,
                 energy_kwh=-1., arrival_net_soc_depletion_pp=-2.),
            dict(departure=pd.Timestamp("2018-01-02 20:00"),
                 arrival=pd.Timestamp("2018-01-02 23:30"), distance_km=20.,
                 energy_kwh=2., arrival_net_soc_depletion_pp=4.),
        ])
        shifted = delay_trip_returns(rows, 90, start="2018-01-02", end="2018-01-03")
        self.assertEqual(len(shifted), 2)
        self.assertEqual(shifted.iloc[0].departure, rows.iloc[0].departure)
        self.assertEqual(shifted.iloc[0].arrival, pd.Timestamp("2018-01-02 01:00"))
        self.assertEqual(shifted.iloc[1].arrival, pd.Timestamp("2018-01-03 01:00"))
        np.testing.assert_array_equal(shifted.energy_kwh, rows.energy_kwh)
        np.testing.assert_array_equal(shifted.arrival_net_soc_depletion_pp,
                                      rows.arrival_net_soc_depletion_pp)

    def test_overlapping_or_touching_return_is_rejected_without_changing_trips(self):
        original = pd.DataFrame([
            dict(departure=pd.Timestamp("2018-01-01 09:00"),
                 arrival=pd.Timestamp("2018-01-01 18:30"), distance_km=10.),
            dict(departure=pd.Timestamp("2018-01-01 19:30"),
                 arrival=pd.Timestamp("2018-01-01 20:30"), distance_km=2.),
        ])
        for minutes in (60, 90):
            with self.subTest(minutes=minutes), self.assertRaisesRegex(ValueError, "next departure"):
                delay_trip_returns(original, minutes)
        self.assertEqual(original.arrival.iloc[0], pd.Timestamp("2018-01-01 18:30"))
        self.assertEqual(len(original), 2)

    def test_zero_delay_remains_exact_and_invalid_minutes_fail_before_run(self):
        rows = pd.DataFrame([dict(departure=pd.Timestamp("2018-01-01 09:00"),
                                  arrival=pd.Timestamp("2018-01-01 17:00"), distance_km=10.)])
        pd.testing.assert_frame_equal(delay_trip_returns(rows, 0), rows)
        for minutes in (-1, 90.0, 1.5, True, float("nan")):
            with self.subTest(minutes=minutes), self.assertRaisesRegex(ValueError, "whole minute"):
                vehicle(return_delay_minutes=minutes)
        shifted = delay_trip_returns(rows, 90)
        with self.assertRaisesRegex(ValueError, "exactly once"):
            delay_trip_returns(shifted, 90)
        empty = delay_trip_returns([], 90)
        self.assertEqual(len(EVSchedule(empty, "2018-01-01", "2018-01-02", {}).trips), 0)
        with self.assertRaisesRegex(ValueError, "return_delay_collision_policy"):
            vehicle(return_delay_collision_policy="merge")


if __name__ == "__main__":
    unittest.main()
