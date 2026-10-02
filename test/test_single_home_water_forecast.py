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

    def test_requires_history_and_full_48_step_quarter_hour_grid(self):
        data = water_data()
        data.source_start = data.flow.index[0] + pd.Timedelta(days=2)
        with self.assertRaisesRegex(ValueError, "seven complete historical"):
            ScenarioWaterForecast(data)
        data = water_data()
        with self.assertRaisesRegex(ValueError, "12-hour horizon"):
            ScenarioWaterForecast(data, horizon=pd.Timedelta(hours=6))
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


if __name__ == "__main__":
    unittest.main()
