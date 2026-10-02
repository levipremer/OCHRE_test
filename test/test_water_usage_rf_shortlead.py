"""Focused checks for the training-only, causal RF scenario correction."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from examples.single_home_hems.water_usage_forecasting.rf_shortlead import (
    RFShortLeadConfig,
    RFShortLeadModel,
    apply_rf_shortlead,
    fit_rf_shortlead,
)
from examples.single_home_hems.water_usage_forecasting.scenario_methods import ScenarioSet


TZ = "America/New_York"


class FakeForest:
    def __init__(self, *, forecast=(0.35, 0.1), **kwargs):
        self.forecast = forecast
        self.features = None
        self.targets = None

    def fit(self, features, targets):
        self.features = np.asarray(features)
        self.targets = np.asarray(targets)
        return self

    def predict(self, features):
        return np.asarray([self.forecast] * len(features))


class RFShortLeadTests(unittest.TestCase):
    def test_training_features_end_before_both_targets_and_cutoff(self):
        index = pd.date_range("2025-11-01", periods=8, freq="15min", tz=TZ)
        training = pd.Series(np.arange(8, dtype=float), index=index)
        fake = FakeForest()
        with patch("sklearn.ensemble.RandomForestRegressor", return_value=fake):
            model = fit_rf_shortlead(training, RFShortLeadConfig(forest_trees=5))
        np.testing.assert_array_equal(fake.features[:, 0], [3, 4, 5])
        np.testing.assert_array_equal(fake.targets, [[4, 5], [5, 6], [6, 7]])
        self.assertEqual(model.training_rows, 3)
        self.assertEqual(model.training_end_exclusive, index[-1] + pd.Timedelta(minutes=15))
        self.assertEqual(model.diagnostics["target_leads_minutes"], [15, 30])

    def test_future_rows_are_ignored_and_early_origin_is_rejected(self):
        origin = pd.Timestamp("2025-11-06 08:00", tz=TZ)
        prior = pd.date_range(end=origin - pd.Timedelta(minutes=15), periods=4, freq="15min")
        history = pd.Series([0.0, 0.5, 0.0, 1.0], index=prior)
        model = RFShortLeadModel(
            estimator=FakeForest(), config=RFShortLeadConfig(),
            training_start=origin - pd.Timedelta(days=1),
            training_end_exclusive=origin, training_rows=100,
        )
        expected = model.predict(history, origin)
        leaked = pd.concat([history, pd.Series(999.0, index=pd.date_range(
            origin, periods=4, freq="15min"
        ))])
        np.testing.assert_array_equal(model.predict(leaked, origin), expected)
        self.assertIsNone(model.predict(history.drop(prior[0]), origin))
        with self.assertRaisesRegex(ValueError, "training cutoff"):
            model.predict(history, origin - pd.Timedelta(minutes=15))

    def test_one_and_two_step_corrections_preserve_weights_and_later_paths(self):
        origin = pd.Timestamp("2025-11-06 08:00", tz=TZ)
        prior = pd.date_range(end=origin - pd.Timedelta(minutes=15), periods=4, freq="15min")
        history = pd.Series([0.0, 0.5, 0.0, 1.0], index=prior)
        model = RFShortLeadModel(
            estimator=FakeForest(), config=RFShortLeadConfig(),
            training_start=origin - pd.Timedelta(days=1),
            training_end_exclusive=origin, training_rows=100,
        )
        base = ScenarioSet(
            values=np.asarray([[0.1, 0.0, 2.0], [0.3, 1.0, 3.0]]),
            weights=np.asarray([0.25, 0.75]),
            candidate_ids=("day_a", "day_b"),
            diagnostics={"origin": origin.isoformat()},
        )
        one = apply_rf_shortlead(base, history, model, steps=1)
        two = apply_rf_shortlead(base, history, model, steps=2)
        np.testing.assert_allclose(one.values[:, 0], [0.2, 0.4])
        np.testing.assert_array_equal(one.values[:, 1:], base.values[:, 1:])
        np.testing.assert_allclose(two.values[:, 1], [0.0, 2 / 15])
        np.testing.assert_array_equal(two.values[:, 2:], base.values[:, 2:])
        np.testing.assert_array_equal(two.weights, base.weights)
        np.testing.assert_array_equal(base.values, [[0.1, 0.0, 2.0], [0.3, 1.0, 3.0]])
        self.assertEqual(two.diagnostics["rf_shortlead_steps_applied"], 2)
        self.assertEqual(two.diagnostics["rf_input_end_exclusive"], origin.isoformat())
        self.assertTrue(two.diagnostics["rf_projection_zero_floor_active"][1])
        np.testing.assert_allclose(
            two.diagnostics["rf_adapted_weighted_mean_l_per_min"],
            two.diagnostics["rf_predicted_l_per_min"], atol=1e-12,
        )
        np.testing.assert_allclose(two.diagnostics["rf_target_mean_error_l_per_min"], [0, 0], atol=1e-12)
        with self.assertRaisesRegex(ValueError, "already corrected"):
            apply_rf_shortlead(two, history, model, steps=1)

    def test_projection_matches_zero_heavy_and_zero_rf_targets(self):
        origin = pd.Timestamp("2025-11-06 08:00", tz=TZ)
        prior = pd.date_range(end=origin - pd.Timedelta(minutes=15), periods=4, freq="15min")
        history = pd.Series([0.0, 0.5, 0.0, 1.0], index=prior)
        model = RFShortLeadModel(
            estimator=FakeForest(forecast=(.05, 0.0)), config=RFShortLeadConfig(),
            training_start=origin - pd.Timedelta(days=1),
            training_end_exclusive=origin, training_rows=100,
        )
        base = ScenarioSet(
            np.asarray([[0.0, 0.2, 3.0], [0.0, 0.0, 2.0], [2.0, 1.0, 1.0]]),
            np.asarray([.6, .2, .2]), ("a", "b", "c"),
            {"origin": origin.isoformat()},
        )
        adapted = apply_rf_shortlead(base, history, model, steps=2)
        np.testing.assert_allclose(adapted.values[:, 0], [0.0, 0.0, .25])
        np.testing.assert_array_equal(adapted.values[:, 1], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(adapted.values[:, 2], base.values[:, 2])
        np.testing.assert_allclose(adapted.weights @ adapted.values[:, :2], [.05, 0], atol=1e-12)

    def test_missing_completed_hour_skips_both_leads(self):
        origin = pd.Timestamp("2025-11-06 08:00", tz=TZ)
        prior = pd.date_range(end=origin - pd.Timedelta(minutes=15), periods=4, freq="15min")
        history = pd.Series([0.0, np.nan, 0.0, 1.0], index=prior)
        model = RFShortLeadModel(
            estimator=FakeForest(), config=RFShortLeadConfig(),
            training_start=origin - pd.Timedelta(days=1),
            training_end_exclusive=origin, training_rows=100,
        )
        base = ScenarioSet(np.asarray([[0.0, 0.1]]), np.asarray([1.0]),
                           ("day_a",), {"origin": origin.isoformat()})
        adapted = apply_rf_shortlead(base, history, model, steps=2)
        np.testing.assert_array_equal(adapted.values, base.values)
        self.assertEqual(adapted.diagnostics["rf_shortlead_steps_applied"], 0)
        self.assertEqual(adapted.diagnostics["rf_shortlead_skip_reason"], "missing_complete_prior_hour")


if __name__ == "__main__":
    unittest.main()
