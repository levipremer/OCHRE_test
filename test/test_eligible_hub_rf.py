"""Check causal issue features against the original plain-history RF study."""

import numpy as np
import pandas as pd

from examples.single_home_hems.forecasting.evaluate_simulated_home_hub import (
    make_supervised,
)
from examples.single_home_hems.single_home_hems.eligible_hub_rf import (
    EligibleHubRFProvider,
)
from examples.single_home_hems.single_home_hems.home_hub_observation import HUB_POWER_KEY


class _HubLagModel:
    def predict(self, features):
        return features["load_lag_1_kw"].to_numpy(float)


class _EligibleLagModel:
    def predict(self, features):
        lag = features["load_lag_1_kw"].to_numpy(float)
        return np.column_stack((lag, lag, 0.5 * lag))


def _provider(seed, ambient, columns):
    return EligibleHubRFProvider(
        _HubLagModel(), _EligibleLagModel(), tuple(columns), seed, ambient,
        "EV_01A", {"inverter_kw": 9.6},
    )


def test_issue_features_match_original_training_features():
    index = pd.date_range("2018-01-01", periods=9 * 96, freq="15min")
    data = pd.DataFrame({
        "hub_non_ev_power_kw": 2 + np.sin(np.arange(len(index)) / 8),
        "ambient_temperature_c": -3 + np.cos(np.arange(len(index)) / 16),
    }, index=index)
    rows = make_supervised(data)
    issue = index[8 * 96]
    provider = _provider(
        data.loc[data.index < issue, "hub_non_ev_power_kw"],
        data["ambient_temperature_c"], rows.feature_sets["history"],
    )
    expected = rows.x.loc[rows.origin == issue, list(rows.feature_sets["history"])].reset_index(drop=True)
    actual = provider._issue_features(issue, provider.seed_history).reset_index(drop=True)
    np.testing.assert_allclose(actual.to_numpy(float), expected.to_numpy(float), atol=1e-12)


def test_issue_uses_completed_live_bin_and_excludes_future_minutes():
    issue = pd.Timestamp("2025-12-01 00:15")
    seed_index = pd.date_range(issue - pd.Timedelta(days=7), issue, freq="15min", inclusive="left")
    seed = pd.Series(1.0, index=seed_index)
    ambient = pd.Series(18.0, index=seed_index)
    provider = _provider(seed, ambient, ("load_lag_1_kw",))
    minute_end = pd.date_range(issue - pd.Timedelta(minutes=14), periods=30, freq="min")
    history = pd.DataFrame({
        HUB_POWER_KEY: np.r_[np.full(15, 2.0), np.full(15, 50.0)],
        "EV_01A Electric Power (kW)": np.zeros(30),
    }, index=minute_end)
    issued = provider.issue(issue, history, pd.Timedelta(hours=8), 9.6)
    assert (issued.intervals["hub_mean_kw"] == 2.0).all()
    assert (issued.intervals["eligible_mean_kw"] == 2.0).all()
    assert provider.history[-1]["latest_live_complete_bin"] == issue - pd.Timedelta(minutes=15)
    assert issued.inputs_ready_at == issue
