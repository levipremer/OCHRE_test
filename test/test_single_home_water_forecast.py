"""Water replay alignment and end-to-end causal MPC scenario forecasts."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from examples.single_home_hems.single_home_hems.forecast import Forecast
from examples.single_home_hems.single_home_hems.water_forecast import (
    DCHouseWaterData, ScenarioWaterForecast,
)
from examples.single_home_hems.water_usage_forecasting.rf_shortlead import RFShortLeadConfig


TZ = "America/New_York"


def water_data():
    index = pd.date_range("2025-10-01", periods=12 * 1440, freq="min", tz=TZ)
    flow = pd.Series(0.0, index=index, name="water_flow_l_per_min")
    for day in range(12):
        start = index[0] + pd.Timedelta(days=day)
        flow.loc[start + pd.Timedelta(hours=6):start + pd.Timedelta(hours=6, minutes=14)] = .2
        flow.loc[start + pd.Timedelta(hours=8):start + pd.Timedelta(hours=8, minutes=14)] = day * .2
    source = pd.Timestamp("2025-10-09", tz=TZ)
    flow.loc[source + pd.Timedelta(hours=6):source + pd.Timedelta(hours=6, minutes=14)] = 2.0
    return DCHouseWaterData(flow, pd.Timestamp("2018-01-01"), source)


def matched_history_data():
    """Give each prior day a distinct six-hour match and intact future path."""
    source = pd.Timestamp("2025-01-30 08:00", tz=TZ)
    index = pd.date_range("2024-12-30", "2025-02-01", freq="min", inclusive="left", tz=TZ)
    flow = pd.Series(0.0, index=index, name="water_flow_l_per_min")
    for days_back in range(1, 31):
        candidate = source - pd.DateOffset(days=days_back)
        flow.loc[candidate - pd.Timedelta(hours=6):candidate - pd.Timedelta(minutes=1)] = (
            1.0 - days_back / 100
        )
        flow.loc[candidate:candidate + pd.Timedelta(hours=12, minutes=-1)] = days_back / 10
    flow.loc[source - pd.Timedelta(hours=6):source - pd.Timedelta(minutes=1)] = 1.0
    return DCHouseWaterData(flow, pd.Timestamp("2018-01-01 08:00"), source)


class WaterForecastTests(unittest.TestCase):
    def test_replay_preserves_local_time_and_volume_across_seasons(self):
        data = water_data()
        index = pd.date_range("2018-01-01 06:00", periods=15, freq="min", tz=TZ)
        replay = data.flow_at(index)
        self.assertTrue(replay.index.equals(index))
        self.assertEqual(float(replay.sum()), 30.0)
        self.assertEqual(data.source_time(index[0]), pd.Timestamp("2025-10-09 06:00", tz=TZ))
        coarse = data.flow_at(index[:1], timestep=pd.Timedelta(minutes=15))
        self.assertEqual(float(coarse.iloc[0]) * 15, replay.sum())

    def test_missing_plant_draws_fail_instead_of_becoming_zero(self):
        data = water_data()
        data.flow.loc[data.source_start] = np.nan
        with self.assertRaisesRegex(ValueError, "missing or unavailable"):
            data.flow_at(pd.date_range(data.simulation_start, periods=10, freq="min"))

    def test_rf_changes_only_first_step_and_uses_frozen_prior_training(self):
        data = water_data()
        base = ScenarioWaterForecast(data, method="daily_budget_high", lookback_days=8)
        rf = ScenarioWaterForecast(data, lookback_days=8, rf_config=RFShortLeadConfig(forest_trees=3))
        timestamp = data.simulation_start + pd.Timedelta(hours=8)
        plain, corrected = base.scenarios_at(timestamp), rf.scenarios_at(timestamp)
        self.assertEqual(plain.values.shape, (8, 48))
        self.assertTrue(plain.diagnostics["morning_condition_active"])
        self.assertTrue((plain.weights > 0).all())
        self.assertGreater(float(np.ptp(plain.weights)), 0)
        np.testing.assert_array_equal(plain.values[:, 1:], corrected.values[:, 1:])
        np.testing.assert_array_equal(plain.weights, corrected.weights)
        self.assertEqual(plain.candidate_ids, corrected.candidate_ids)
        self.assertEqual(rf.rf_model.training_end_exclusive, data.source_start)
        self.assertAlmostEqual(
            corrected.weights @ corrected.values[:, 0],
            corrected.diagnostics["rf_predicted_l_per_min"][0], places=12,
        )
        self.assertEqual(corrected.diagnostics["optimization_probability"], 1 / 8)
        self.assertEqual(corrected.diagnostics["optimization_probability_policy"], "uniform")
        # Even accidentally supplied future observations cannot affect the issue.
        origin = data.source_time(timestamp)
        rf._interval_flow.loc[rf._interval_flow.index >= origin] = 999.0
        repeated = rf.scenarios_at(timestamp)
        np.testing.assert_array_equal(repeated.values, corrected.values)
        np.testing.assert_array_equal(repeated.weights, corrected.weights)

    def test_requires_history_and_whole_quarter_hour_horizon(self):
        data = water_data()
        data.source_start = data.flow.index[0] + pd.Timedelta(days=2)
        with self.assertRaisesRegex(ValueError, "seven complete historical"):
            ScenarioWaterForecast(data)
        data = water_data()
        with self.assertRaisesRegex(ValueError, "positive whole-step horizon"):
            ScenarioWaterForecast(data, horizon=pd.Timedelta(minutes=10))
        six_hour = ScenarioWaterForecast(data, method="daily_budget_high",
                                         horizon=pd.Timedelta(hours=6))
        self.assertEqual(six_hour.scenarios_at(data.simulation_start).values.shape[1], 24)
        forecast = ScenarioWaterForecast(data, method="daily_budget_high")
        with self.assertRaisesRegex(ValueError, "quarter hours"):
            forecast.scenarios_at(data.simulation_start + pd.Timedelta(minutes=1))

    def test_csv_and_forecast_attachment_keep_units_and_source_time(self):
        data = water_data()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "flow.csv"
            data.flow.rename_axis("timestamp").to_csv(path)
            loaded = DCHouseWaterData.from_csv(path, data.simulation_start, "2025-10-09")
            np.testing.assert_allclose(loaded.flow, data.flow, rtol=1e-14, atol=0)
            self.assertEqual(loaded.source_start, data.source_start)
            with self.assertRaisesRegex(ValueError, "same local time of day"):
                DCHouseWaterData.from_csv(path, data.simulation_start, "2025-10-09 01:00")
        provider = ScenarioWaterForecast(data, method="daily_budget_high")
        index = pd.date_range(data.simulation_start, periods=2, freq="min")
        forecast = Forecast.from_schedule(pd.DataFrame(index=index), lambda _: .15, water_scenarios=provider)
        self.assertIs(forecast.water_scenarios, provider)

    def test_six_hour_rmse_volume_methods_select_exact_equal_weight_paths_causally(self):
        data = matched_history_data()
        for count in (5, 7, 10):
            with self.subTest(count=count):
                method = f"rmse_volume_6h_k{count}"
                provider = ScenarioWaterForecast(data, method=method, lookback_days=30)
                chosen = provider.scenarios_at(data.simulation_start)
                expected_ids = tuple(
                    str((data.source_start - pd.DateOffset(days=days_back)).date())
                    for days_back in range(1, count + 1)
                )
                self.assertEqual(chosen.candidate_ids, expected_ids)
                self.assertEqual(chosen.values.shape, (count, 48))
                np.testing.assert_allclose(chosen.weights, np.full(count, 1 / count))
                np.testing.assert_allclose(
                    chosen.values,
                    np.repeat(np.arange(1, count + 1, dtype=float)[:, None] / 10, 48, axis=1),
                )
                self.assertIsNone(provider.rf_model)
                self.assertEqual(chosen.diagnostics["selection_method"], "rmse_volume")
                self.assertEqual(chosen.diagnostics["hours_preceding"], 6)
                self.assertEqual(chosen.diagnostics["volume_weight"], 0.5)
                self.assertEqual(chosen.diagnostics["optimization_probability"], 1 / count)
                self.assertEqual(provider.metadata["requested_scenario_count"], count)

                # The provider must not use readings from the forecast horizon.
                origin = data.source_start
                provider._interval_flow.loc[provider._interval_flow.index >= origin] = 999.0
                repeated = provider.scenarios_at(data.simulation_start)
                self.assertEqual(repeated.candidate_ids, chosen.candidate_ids)
                np.testing.assert_array_equal(repeated.values, chosen.values)

    def test_six_hour_rmse_volume_requires_requested_path_count(self):
        data = water_data()
        with self.assertRaisesRegex(ValueError, "at least 10 complete historical paths"):
            ScenarioWaterForecast(data, method="rmse_volume_6h_k10", lookback_days=8)

    def test_six_hour_rmse_volume_rf_changes_only_first_selected_lead(self):
        data = matched_history_data()
        for count in (5, 7, 10):
            with self.subTest(count=count):
                base_method = f"rmse_volume_6h_k{count}"
                base = ScenarioWaterForecast(data, method=base_method)
                rf = ScenarioWaterForecast(
                    data, method=f"{base_method}_rf_1",
                    rf_config=RFShortLeadConfig(forest_trees=3),
                )
                plain = base.scenarios_at(data.simulation_start)
                corrected = rf.scenarios_at(data.simulation_start)
                self.assertEqual(corrected.candidate_ids, plain.candidate_ids)
                np.testing.assert_array_equal(corrected.weights, plain.weights)
                np.testing.assert_array_equal(corrected.values[:, 1:], plain.values[:, 1:])
                self.assertEqual(corrected.diagnostics["rf_shortlead_steps_applied"], 1)
                self.assertFalse(np.array_equal(corrected.values[:, 0], plain.values[:, 0]))
                self.assertAlmostEqual(
                    corrected.values[:, 0].mean(),
                    corrected.diagnostics["rf_predicted_l_per_min"][0],
                )
                self.assertEqual(corrected.diagnostics["optimization_probability"], 1 / count)
                self.assertEqual(rf.metadata["requested_scenario_count"], count)
                self.assertIn("RF first-step", rf.metadata["conditioning_weight_policy"])
                self.assertEqual(rf.rf_model.training_end_exclusive, data.source_start)

    def test_study_choice_arms_apply_rf_to_first_step_at_variable_horizon(self):
        data = matched_history_data()
        for method in (
            "day_type", "rmse_volume_6h", "morning_progress", "tail_anchored", "recent",
        ):
            with self.subTest(method=method):
                provider = ScenarioWaterForecast(
                    data, method=f"{method}_k5_rf_1",
                    horizon=pd.Timedelta(hours=3),
                    rf_config=RFShortLeadConfig(forest_trees=3),
                )
                chosen = provider.scenarios_at(data.simulation_start)
                self.assertEqual(chosen.values.shape, (5, 12))
                np.testing.assert_allclose(chosen.weights, np.full(5, .2))
                self.assertEqual(chosen.diagnostics["rf_shortlead_steps_applied"], 1)
                self.assertEqual(chosen.diagnostics["optimization_probability"], .2)
                self.assertEqual(provider.rf_model.training_end_exclusive, data.source_start)
                if method == "tail_anchored":
                    self.assertEqual(chosen.diagnostics["tail_ranking_hours"], 3.0)
                    self.assertEqual(chosen.diagnostics["tail_ranking_policy"], "forecast_horizon")

    def test_horizon_sweep_sizes_match_controller_grid(self):
        data = matched_history_data()
        for hours in (3, 4, 6, 8, 12, 16, 24):
            with self.subTest(hours=hours):
                provider = ScenarioWaterForecast(
                    data, method="rmse_volume_6h_k5",
                    horizon=pd.Timedelta(hours=hours),
                )
                chosen = provider.scenarios_at(data.simulation_start)
                self.assertEqual(chosen.values.shape, (5, hours * 4))
                self.assertEqual(provider.metadata["horizon_hours"], float(hours))


if __name__ == "__main__":
    unittest.main()
