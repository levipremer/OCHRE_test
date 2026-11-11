"""Runtime/training alignment and causal circuit-history forecasting."""

import numpy as np
import pandas as pd
import pytest

from examples.single_home_hems.forecasting import evaluate_simulated_hub_submeter_rf as study
from examples.single_home_hems.single_home_hems.home_hub_observation import HUB_POWER_KEY
from examples.single_home_hems.single_home_hems.submeter_hub_rf import (
    CIRCUIT_COLUMNS, SubmeterHubRFProvider, _project,
)


class _HubHistoryModel:
    def predict(self, features):
        self.last_features = features.copy()
        return features.load_lag_1_kw.to_numpy(float)


class _EligibleHistoryModel:
    def predict(self, features):
        raw = features.load_lag_1_kw.to_numpy(float)
        return np.column_stack((raw + 1, raw + 2, 0.5 * raw))


def _observations():
    index = pd.date_range("2018-01-01", periods=9 * 96, freq="15min")
    t = np.arange(len(index), dtype=float)
    circuits = pd.DataFrame({
        "water_heater_kw": 0.4 + 0.2 * np.sin(t / 9) ** 2,
        "outdoor_heat_pump_kw": 1.0 + np.cos(t / 15) ** 2,
        "indoor_fan_kw": 0.2 + 0.1 * np.sin(t / 15) ** 2,
    }, index=index)
    data = pd.DataFrame({"hub_non_ev_power_kw": circuits.sum(axis=1) + 0.3,
                         "ambient_temperature_c": -3 + np.cos(t / 96)}, index=index)
    rows = study.make_supervised(data, circuits)
    return data, circuits, rows, index[8 * 96]


def _provider(data, circuits, rows):
    return SubmeterHubRFProvider(
        _HubHistoryModel(), _EligibleHistoryModel(), rows.feature_sets[study.METHODS[1]],
        data.hub_non_ev_power_kw.copy(), circuits.copy(), data.ambient_temperature_c.copy(),
        "EV_01A", {"inverter_kw": 9.6},
    )


def test_runtime_features_match_all_100_evaluated_features_for_both_horizons():
    data, circuits, rows, issue = _observations()
    provider = _provider(data, circuits, rows)
    expected = rows.x.loc[rows.origin == issue, list(provider.feature_columns)].reset_index(drop=True)
    for hours, steps in ((8, 32), (20, 80)):
        forecast = provider.issue(issue, pd.DataFrame(), pd.Timedelta(hours=hours), 9.6)
        features = provider.hub_model.last_features
        assert list(features) == list(expected)
        np.testing.assert_allclose(features.to_numpy(float), expected.iloc[:steps].to_numpy(float),
                                   rtol=0, atol=1e-12)
        assert len(forecast.intervals) == steps
        assert forecast.intervals.index[-1] == issue + (steps - 1) * study.INTERVAL
        assert provider.history[-1]["direct_rf_steps"] == steps
        assert provider.history[-1]["tail_source_day"] is None


def _live(issue):
    ends = pd.date_range(issue - pd.Timedelta(minutes=14), periods=30, freq="min")
    return pd.DataFrame({HUB_POWER_KEY: 6.0, "EV_01A Electric Power (kW)": 1.0,
                         "water_heater_kw": 0.7, "outdoor_heat_pump_kw": 2.0,
                         "indoor_fan_kw": 0.2}, index=ends)


def test_live_minute_end_alignment_and_future_seed_weather_circuits_do_not_leak():
    data, circuits, rows, issue = _observations()
    provider = _provider(data, circuits, rows)
    live = _live(issue)
    baseline = provider.issue(issue, live, pd.Timedelta(hours=20), 9.6)
    features = provider.hub_model.last_features.copy()
    assert features.load_lag_1_kw.eq(5).all()
    assert features.water_heater_lag_1_kw.eq(0.7).all()
    assert provider.history[-1]["latest_live_complete_bin"] == issue - study.INTERVAL
    assert baseline.inputs_ready_at == issue

    changed_data, changed_circuits = data.copy(), circuits.copy()
    changed_data.loc[changed_data.index >= issue] += 1000
    changed_circuits.loc[changed_circuits.index >= issue] += 1000
    altered = _provider(changed_data, changed_circuits, rows)
    changed_live = live.copy()
    changed_live.loc[changed_live.index > issue] = np.nan
    future_changed = altered.issue(issue, changed_live, pd.Timedelta(hours=20), 9.6)
    pd.testing.assert_frame_equal(provider.hub_model.last_features, altered.hub_model.last_features)
    pd.testing.assert_frame_equal(baseline.intervals, future_changed.intervals)

    changed_live = live.copy()
    changed_live.loc[issue, "water_heater_kw"] += 1
    changed_live.loc[issue, HUB_POWER_KEY] += 1
    altered.issue(issue, changed_live, pd.Timedelta(hours=8), 9.6)
    np.testing.assert_allclose(altered.hub_model.last_features.water_heater_lag_1_kw,
                               0.7 + 1 / 15, rtol=0, atol=1e-12)


def test_missing_live_circuits_or_minutes_are_rejected_without_seed_fallback():
    data, circuits, rows, issue = _observations()
    provider = _provider(data, circuits, rows)
    live = _live(issue)
    for circuit in CIRCUIT_COLUMNS:
        with pytest.raises(ValueError, match="lacks"):
            provider.issue(issue, live.drop(columns=circuit), pd.Timedelta(hours=8), 9.6)
        missing = live.copy()
        missing.loc[issue, circuit] = np.nan
        with pytest.raises(ValueError, match="finite and nonnegative"):
            provider.issue(issue, missing, pd.Timedelta(hours=8), 9.6)
    with pytest.raises(ValueError, match="incomplete completed"):
        provider.issue(issue, live.drop(index=issue), pd.Timedelta(hours=8), 9.6)
    bad = live.copy()
    bad.loc[issue, "outdoor_heat_pump_kw"] = 20
    with pytest.raises(ValueError, match="exceed"):
        provider.issue(issue, bad, pd.Timedelta(hours=8), 9.6)


def test_eligibility_projection_preserves_legacy_mean_minimum_and_inverter_caps():
    hub = np.array([0.0, 3.0, 11.0, 6.0])
    eligible = np.array([[4, 1, 2], [12, 11, 20], [20, 14, 13], [1, 2, 3]], dtype=float)
    expected = np.array([[0, 0, 0], [3, 3, 3], [11, 9.6, 9.6], [1, 1, 1]])
    np.testing.assert_array_equal(_project(hub, eligible, 9.6), expected)
    bad = eligible.copy()
    bad[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite complete"):
        _project(hub, bad, 9.6)


def test_provider_requires_complete_seven_day_histories_and_matching_inverter():
    data, circuits, rows, issue = _observations()
    provider = _provider(data, circuits, rows)
    provider.seed_circuit_history.loc[issue - pd.Timedelta(days=7), "indoor_fan_kw"] = np.nan
    with pytest.raises(ValueError, match="complete causal seven-day"):
        provider.issue(issue, pd.DataFrame(), pd.Timedelta(hours=20), 9.6)
    with pytest.raises(ValueError, match="different inverter"):
        provider.issue(issue, pd.DataFrame(), pd.Timedelta(hours=8), 7.2)
