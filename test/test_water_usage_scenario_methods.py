"""Synthetic checks for causal, intact water-use scenarios."""

import unittest

import numpy as np
import pandas as pd

from examples.single_home_hems.water_usage_forecasting.scenario_methods import (
    ScenarioConfig,
    apply_persistence,
    generate_scenarios,
)


TZ = "America/New_York"


def history_through(origin: pd.Timestamp, days: int = 4) -> pd.Series:
    index = pd.date_range(
        origin - pd.Timedelta(days=days), origin, inclusive="left", freq="15min"
    )
    return pd.Series(0.0, index=index)


def scenarios_at(history: pd.Series, origin: pd.Timestamp, **kwargs):
    config = ScenarioConfig(**kwargs)
    future = pd.date_range(origin, periods=48, freq="15min")
    return generate_scenarios(history, origin, future, config)


class HistoricalScenarioTests(unittest.TestCase):
    def test_complete_analog_keeps_draw_cluster_and_excludes_future_history(self):
        origin = pd.Timestamp("2025-11-06 08:00", tz=TZ)
        history = history_through(origin, days=3)
        for start, amplitude in ((pd.Timestamp("2025-11-04 09:00", tz=TZ), 1.0),
                                 (pd.Timestamp("2025-11-05 10:00", tz=TZ), 2.0)):
            history.loc[start:start + pd.Timedelta(minutes=30)] = amplitude
        base = scenarios_at(history, origin, lookback_days=2)
        future_rows = pd.Series(999.0, index=pd.date_range(origin, periods=48, freq="15min"))
        with_future = scenarios_at(pd.concat([history, future_rows]), origin, lookback_days=2)
        self.assertEqual(base.candidate_ids, ("2025-11-05", "2025-11-04"))
        np.testing.assert_array_equal(base.values, with_future.values)
        self.assertEqual(with_future.diagnostics["history_future_rows_ignored"], 48)
        np.testing.assert_allclose(base.weights, [0.5, 0.5])
        self.assertEqual(base.values.shape, (2, 48))
        # Three adjacent 15-minute draw bins remain adjacent in each scenario.
        self.assertEqual(int(np.count_nonzero(base.values[0])), 3)
        self.assertEqual(int(np.count_nonzero(base.values[1])), 3)

    def test_morning_budget_downweights_another_large_draw_without_discarding_it(self):
        origin = pd.Timestamp("2025-11-06 08:00", tz=TZ)
        history = history_through(origin, days=3)
        early_day = pd.Timestamp("2025-11-04 06:00", tz=TZ)
        late_day = pd.Timestamp("2025-11-05 08:00", tz=TZ)
        history.loc[early_day:early_day + pd.Timedelta(hours=2) - pd.Timedelta(minutes=15)] = 2.0
        history.loc[late_day:late_day + pd.Timedelta(hours=4) - pd.Timedelta(minutes=15)] = 1.0
        today_early = pd.Timestamp("2025-11-06 06:00", tz=TZ)
        history.loc[today_early:origin - pd.Timedelta(minutes=15)] = 2.0

        result = scenarios_at(
            history, origin, lookback_days=2, morning_policy="remaining_budget",
            budget_scale_l=10,
        )
        self.assertEqual(result.candidate_ids, ("2025-11-05", "2025-11-04"))
        self.assertTrue(result.diagnostics["morning_condition_active"])
        self.assertAlmostEqual(result.diagnostics["current_morning_volume_l"], 240)
        self.assertAlmostEqual(result.diagnostics["typical_morning_volume_l"], 240)
        self.assertAlmostEqual(result.diagnostics["remaining_budget_l"], 0)
        np.testing.assert_allclose(result.diagnostics["candidate_remaining_morning_l"], [240, 0])
        self.assertGreater(result.weights[1], result.weights[0])
        self.assertGreater(result.weights[0], 0)
        self.assertAlmostEqual(result.weights.sum(), 1)

    def test_morning_condition_does_not_suppress_tomorrow_morning(self):
        origin = pd.Timestamp("2025-11-06 20:00", tz=TZ)
        history = history_through(origin, days=3)
        for policy in ("remaining_budget", "remaining_budget_high_only"):
            with self.subTest(policy=policy):
                result = scenarios_at(history, origin, lookback_days=2, morning_policy=policy)
                self.assertFalse(result.diagnostics["morning_condition_active"])
                self.assertEqual(result.diagnostics["morning_condition_reason"], "origin_outside_morning")
                np.testing.assert_allclose(result.weights, [0.5, 0.5])

    def test_high_only_conditions_after_most_morning_use_but_not_before(self):
        origin = pd.Timestamp("2025-11-06 08:00", tz=TZ)
        history = history_through(origin, days=3)
        early_day = pd.Timestamp("2025-11-04 06:00", tz=TZ)
        late_day = pd.Timestamp("2025-11-05 08:00", tz=TZ)
        history.loc[early_day:early_day + pd.Timedelta(hours=2) - pd.Timedelta(minutes=15)] = 2.0
        history.loc[late_day:late_day + pd.Timedelta(hours=4) - pd.Timedelta(minutes=15)] = 1.0
        today_early = pd.Timestamp("2025-11-06 06:00", tz=TZ)

        high = history.copy()
        high.loc[today_early:origin - pd.Timedelta(minutes=15)] = 2.0
        conditional = scenarios_at(high, origin, lookback_days=2,
                                   morning_policy="remaining_budget_high_only")
        unconditional = scenarios_at(high, origin, lookback_days=2,
                                     morning_policy="remaining_budget")
        self.assertTrue(conditional.diagnostics["morning_condition_active"])
        self.assertTrue(conditional.diagnostics["morning_progress_gate_passed"])
        self.assertAlmostEqual(conditional.diagnostics["morning_progress_ratio"], 1.0)
        self.assertGreater(conditional.weights[1], conditional.weights[0])
        np.testing.assert_allclose(conditional.weights, unconditional.weights)

        # Future readings supplied by mistake are ignored for both the gate
        # and scenario selection.
        future_rows = pd.Series(999.0, index=pd.date_range(origin, periods=48, freq="15min"))
        leaked = scenarios_at(pd.concat([high, future_rows]), origin, lookback_days=2,
                              morning_policy="remaining_budget_high_only")
        np.testing.assert_array_equal(leaked.values, conditional.values)
        np.testing.assert_array_equal(leaked.weights, conditional.weights)

        low = history.copy()
        low.loc[today_early:origin - pd.Timedelta(minutes=15)] = 0.5
        gated = scenarios_at(low, origin, lookback_days=2,
                             morning_policy="remaining_budget_high_only")
        plain = scenarios_at(low, origin, lookback_days=2, morning_policy="none")
        budget_without_gate = scenarios_at(low, origin, lookback_days=2,
                                           morning_policy="remaining_budget")
        self.assertFalse(gated.diagnostics["morning_condition_active"])
        self.assertEqual(gated.diagnostics["morning_condition_reason"], "below_progress_threshold")
        self.assertFalse(gated.diagnostics["morning_progress_gate_passed"])
        self.assertAlmostEqual(gated.diagnostics["current_morning_volume_l"], 60)
        self.assertAlmostEqual(gated.diagnostics["morning_progress_threshold_l"], 180)
        np.testing.assert_array_equal(gated.weights, plain.weights)
        self.assertFalse(np.allclose(gated.weights, budget_without_gate.weights))

    def test_one_or_two_step_persistence_and_active_gate(self):
        origin = pd.Timestamp("2025-11-06 08:00", tz=TZ)
        history = history_through(origin, days=2)
        history.loc[origin - pd.Timedelta(minutes=15)] = 1.2
        plain = scenarios_at(history, origin, lookback_days=1)
        one = scenarios_at(history, origin, lookback_days=1, persistence_steps=1)
        two = scenarios_at(history, origin, lookback_days=1, persistence_steps=2)
        active = scenarios_at(history, origin, lookback_days=1,
                              persistence_steps=2, persistence_mode="active_only")
        np.testing.assert_allclose(one.values[:, 0], 1.2)
        np.testing.assert_allclose(one.values[:, 1:], plain.values[:, 1:])
        np.testing.assert_allclose(two.values[:, :2], 1.2)
        np.testing.assert_array_equal(two.values, active.values)
        self.assertEqual(active.diagnostics["persistence_steps_applied"], 2)

        history.loc[origin - pd.Timedelta(minutes=15)] = 0.2
        gated = scenarios_at(history, origin, lookback_days=1,
                             persistence_steps=2, persistence_mode="active_only")
        np.testing.assert_array_equal(gated.values, plain.values)
        self.assertEqual(gated.diagnostics["persistence_steps_applied"], 0)

    def test_decay_blends_leads_without_changing_base_scenarios_or_weights(self):
        origin = pd.Timestamp("2025-11-06 08:00", tz=TZ)
        history = history_through(origin, days=2)
        history.loc[origin - pd.Timedelta(minutes=15)] = 1.2
        base = scenarios_at(history, origin, lookback_days=1)
        base_values = base.values.copy()
        config = ScenarioConfig(lookback_days=1, persistence_steps=2,
                                persistence_strengths=(0.2, 0.06))
        adapted = apply_persistence(base, history, config)
        direct = scenarios_at(history, origin, lookback_days=1,
                              persistence_steps=2, persistence_strengths=(0.2, 0.06))
        np.testing.assert_allclose(adapted.values[:, 0], base_values[:, 0] * 0.8 + 1.2 * 0.2)
        np.testing.assert_allclose(adapted.values[:, 1], base_values[:, 1] * 0.94 + 1.2 * 0.06)
        np.testing.assert_array_equal(adapted.values[:, 2:], base_values[:, 2:])
        np.testing.assert_array_equal(base.values, base_values)
        np.testing.assert_array_equal(adapted.values, direct.values)
        np.testing.assert_array_equal(adapted.weights, base.weights)
        self.assertEqual(adapted.candidate_ids, base.candidate_ids)
        self.assertEqual(adapted.diagnostics["persistence_strengths_applied"], [0.2, 0.06])
        with self.assertRaises(ValueError):
            apply_persistence(adapted, history, config)

    def test_missing_latest_completed_bin_does_not_reuse_stale_flow(self):
        origin = pd.Timestamp("2025-11-06 08:00", tz=TZ)
        history = history_through(origin, days=2)
        history.loc[origin - pd.Timedelta(minutes=30)] = 2.0
        history.loc[origin - pd.Timedelta(minutes=15)] = np.nan
        base = scenarios_at(history, origin, lookback_days=1)
        adapted = apply_persistence(base, history, ScenarioConfig(
            lookback_days=1, persistence_steps=2,
        ))
        np.testing.assert_array_equal(adapted.values, base.values)
        self.assertEqual(adapted.diagnostics["persistence_steps_applied"], 0)
        self.assertIsNone(adapted.diagnostics["persistence_last_flow_l_per_min"])
        self.assertEqual(adapted.diagnostics["persistence_skip_reason"], "missing_latest_completed_bin")

    def test_missing_and_ambiguous_historical_paths_are_not_candidates(self):
        origin = pd.Timestamp("2025-11-04 00:00", tz=TZ)
        history = history_through(origin, days=4)
        result = scenarios_at(history, origin, lookback_days=3)
        self.assertEqual(result.candidate_ids, ("2025-11-03", "2025-11-01"))
        self.assertEqual(result.diagnostics["candidate_exclusion_counts"]["clock_transition"], 1)

        history.loc[pd.Timestamp("2025-11-03 04:00", tz=TZ)] = np.nan
        missing = scenarios_at(history, origin, lookback_days=3)
        self.assertEqual(missing.candidate_ids, ("2025-11-01",))
        self.assertEqual(missing.diagnostics["candidate_exclusion_counts"]["missing_observations"], 1)

    def test_nonexistent_spring_clock_path_is_not_a_candidate(self):
        origin = pd.Timestamp("2025-03-11 00:00", tz=TZ)
        history = history_through(origin, days=4)
        result = scenarios_at(history, origin, lookback_days=3)
        self.assertEqual(result.candidate_ids, ("2025-03-10", "2025-03-08"))
        self.assertEqual(result.diagnostics["candidate_exclusion_counts"]["clock_transition"], 1)

    def test_same_weekday_falls_back_to_daily_when_no_weekly_candidate_exists(self):
        origin = pd.Timestamp("2025-11-06 08:00", tz=TZ)
        history = history_through(origin, days=4)
        result = scenarios_at(history, origin, lookback_days=3, candidate_policy="same_weekday")
        self.assertEqual(len(result.values), 3)
        self.assertEqual(result.diagnostics["effective_candidate_policy"], "daily")
        self.assertTrue(result.diagnostics["same_weekday_fallback_to_daily"])

    def test_invalid_future_and_config_rejected(self):
        origin = pd.Timestamp("2025-11-06 08:00", tz=TZ)
        history = history_through(origin, days=2)
        with self.assertRaises(ValueError):
            generate_scenarios(history, origin,
                               pd.date_range(origin, periods=47, freq="15min"), ScenarioConfig())
        with self.assertRaises(ValueError):
            ScenarioConfig(persistence_steps=49)
        with self.assertRaises(ValueError):
            ScenarioConfig(morning_start_hour=12, morning_end_hour=6)
        with self.assertRaises(ValueError):
            ScenarioConfig(persistence_steps=2, persistence_strengths=(0.2,))
        with self.assertRaises(ValueError):
            ScenarioConfig(persistence_steps=2, persistence_strengths=(0.2, 1.1))
        with self.assertRaises(ValueError):
            ScenarioConfig(morning_progress_fraction=1.1)


if __name__ == "__main__":
    unittest.main()
