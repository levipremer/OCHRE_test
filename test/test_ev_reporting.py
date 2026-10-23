"""Boundary inventory and final-ensemble forecast reporting regressions."""
import unittest
import numpy as np
import pandas as pd
from test.test_ev_v2h import schedule
from examples.single_home_hems.single_home_hems.ev_forecast_metrics import score_forecasts, _energy_bins, _energy
from examples.single_home_hems.single_home_hems.ev_scenarios import persistent_prefix
from examples.single_home_hems.single_home_hems.unified import baseline_economics


class EVReportingTests(unittest.TestCase):
    def test_batched_energy_matches_half_open_reference(self):
        path = schedule([(-10, 0, 2), (1, 7, 3), (8, 15, 4),
                         (16, 29, 0), (30, 37, 5), (38, 45, 6)])
        edges = path.start + pd.to_timedelta([0, 15, 30, 37, 45], unit="min")
        expected = [_energy(path, a, b, .2) for a, b in zip(edges[:-1], edges[1:])]
        np.testing.assert_allclose(_energy_bins(path, edges, .2), expected, rtol=1e-14)
        np.testing.assert_array_equal(_energy_bins(schedule(), edges, .2), 0)

    def score(self, actual, paths, minutes=60, end_minutes=None):
        t = actual.start
        return score_forecasts([(t, pd.Timedelta(minutes=minutes), paths)],
            actual, .2, 15, t, t+pd.Timedelta(minutes=end_minutes or minutes))

    def test_inventory_boundaries_and_missing_archived_values(self):
        frame = pd.DataFrame(dict(home_id="h", device="EV_v",
            case=["all_baseline", "non_v2h_twin", "advanced"],
            cost_usd=[10., 9., 8.], reference_degradation_cost_usd=1.,
            initial_stored_energy_kwh=[20., 22., 18.], final_stored_energy_kwh=[20., 21., 19.],
            initial_energy_value_usd_per_kwh=.1, final_energy_value_usd_per_kwh=.3))
        result = baseline_economics(frame)
        np.testing.assert_allclose(result.energy_inventory_adjustment_usd, [0, .1, -.1])
        np.testing.assert_allclose(result.energy_adjusted_net_savings_usd, [0, 1.1, 1.9])
        np.testing.assert_allclose(result.net_savings_usd, [0, 1, 2])
        self.assertTrue(baseline_economics(frame.drop(columns="initial_stored_energy_kwh")).energy_adjusted_net_savings_usd.isna().all())
        frame.loc[2, "final_energy_value_usd_per_kwh"] = .4
        with self.assertRaisesRegex(ValueError, "common boundary"):
            baseline_economics(frame)

    def test_known_error_and_scenario_error_not_error_of_mean(self):
        actual = schedule([(15, 30, 10)])
        _, leads, scores = self.score(actual, (actual, schedule()))
        for key, expected in dict(brier_home=.0625, false_home=.125, false_away=0,
            arrival_energy_mae_kwh=.25, horizon_energy_mae_kwh=1, energy_coverage=1,
            energy_sharpness_kwh=2, next_trip_energy_mae_kwh=1, next_trip_missing_fraction=.5).items():
            self.assertAlmostEqual(scores["forecast_"+key], expected, msg=key)
        self.assertEqual(len(leads), 4)
        _, _, scores = self.score(actual, (schedule(), schedule([(15, 30, 20)])))
        self.assertEqual(scores["forecast_horizon_energy_mae_kwh"], 2)

    def test_final_prefix_and_no_scenarios(self):
        actual = schedule([(15, 30, 10), (60, 75, 5)])
        modified = persistent_prefix(actual, actual.start, pd.Timedelta(minutes=90),
            pd.Timedelta(minutes=45), True, pd.Timedelta(minutes=1))
        _, _, exact = self.score(actual, (actual,), minutes=90)
        _, _, prefixed = self.score(actual, (modified,), minutes=90)
        self.assertEqual(exact["forecast_brier_home"], 0)
        self.assertEqual(prefixed["forecast_brier_home"], 0)  # matching state: no override
        mismatched = schedule([(-5, 30, 10), (60, 75, 5)])
        modified = persistent_prefix(mismatched, actual.start, pd.Timedelta(minutes=90),
            pd.Timedelta(minutes=45), True, pd.Timedelta(minutes=1))
        _, _, prefixed = self.score(actual, (modified,), minutes=90)
        self.assertGreater(prefixed["forecast_brier_home"], 0)
        rows, leads, scores = self.score(actual, ())
        self.assertTrue(rows.no_scenarios_selected.all())
        self.assertTrue(leads.empty)
        self.assertEqual(scores["forecast_no_scenarios_count"], 1)
        self.assertTrue(np.isnan(scores["forecast_energy_coverage"]))

    def test_next_new_trip_and_missing_truth(self):
        _, _, scores = self.score(schedule(), (schedule([(15, 30, 10)]),))
        self.assertTrue(np.isnan(scores["forecast_next_trip_energy_mae_kwh"]))
        self.assertEqual(scores["forecast_next_trip_false_prediction_fraction"], 1)
        actual = schedule([(-10, 10, 100), (20, 30, 10)])
        predicted = schedule([(-10, 10, 5), (20, 30, 10)])
        _, _, scores = self.score(actual, (predicted,))
        self.assertEqual(scores["forecast_next_trip_energy_mae_kwh"], 0)
        self.assertGreater(scores["forecast_horizon_energy_mae_kwh"], 0)

    def test_phase_masks_partial_horizons_and_empty_history(self):
        actual = schedule()
        t, horizon = actual.start, pd.Timedelta(hours=1)
        issued = [(t-pd.Timedelta(minutes=15), horizon, (actual,)),
                  (t, horizon, (actual,)), (t+pd.Timedelta(minutes=30), horizon, (actual,))]
        rows, leads, scores = score_forecasts(issued, actual, .2, 15, t, t+pd.Timedelta(minutes=30))
        self.assertEqual((len(rows), len(leads), scores["forecast_full_horizon_count"]), (1, 2, 0))
        self.assertTrue(np.isnan(scores["forecast_horizon_energy_mae_kwh"]))
        _, _, scores = score_forecasts([], actual, .2, 15, t, t+horizon)
        self.assertEqual(scores["forecast_issuance_count"], 0)


if __name__ == "__main__":
    unittest.main()
