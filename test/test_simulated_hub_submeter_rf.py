"""Causality and grid checks for the isolated 20-hour submeter RF study."""

import numpy as np
import pandas as pd
import pytest

from examples.single_home_hems.forecasting import evaluate_simulated_home_hub as original
from examples.single_home_hems.forecasting import evaluate_simulated_hub_submeter_rf as study


def observations(days=10):
    index = pd.date_range("2018-01-07", periods=days * 96, freq="15min", name="timestamp")
    t = np.arange(len(index), dtype=float)
    circuits = pd.DataFrame({
        "water_heater_kw": 0.3 + 0.2 * np.sin(t / 7) ** 2,
        "outdoor_heat_pump_kw": 1 + 0.5 * np.sin(t / 13) ** 2,
        "indoor_fan_kw": 0.2 + 0.1 * np.cos(t / 13) ** 2,
    }, index=index)
    non_ev = circuits.sum(axis=1) + 0.1
    ev = np.where(t % 96 < 8, 7.0, 0.0)
    data = pd.DataFrame({
        "home_hub_power_kw": non_ev + ev, "ev_power_kw": ev,
        "hub_non_ev_power_kw": non_ev, "ambient_temperature_c": -5 + np.cos(t / 96),
    }, index=index)
    return data, circuits


def test_aggregate_feature_definition_exactly_matches_original_rf():
    data, circuits = observations()
    reference = original.make_supervised(data)
    rebuilt = study.make_supervised(data, circuits, horizon_steps=32)
    pd.testing.assert_frame_equal(
        rebuilt.x.loc[:, list(rebuilt.feature_sets[study.METHODS[0]])],
        reference.x.loc[:, list(reference.feature_sets["history"])],
    )
    np.testing.assert_array_equal(rebuilt.y, reference.y)
    assert rebuilt.origin.equals(reference.origin)
    assert rebuilt.target.equals(reference.target)
    assert len(rebuilt.feature_sets[study.METHODS[0]]) == 31
    assert len(rebuilt.feature_sets[study.METHODS[1]]) == 100


def test_current_and_future_circuit_load_and_weather_cannot_change_issued_features():
    data, circuits = observations()
    issue = data.index[7 * 96 + 8]
    before = study.make_supervised(data, circuits)
    changed_data, changed_circuits = data.copy(), circuits.copy()
    current_and_future = data.index >= issue
    changed_circuits.loc[current_and_future] += 100
    changed_data.loc[current_and_future, "home_hub_power_kw"] += 20
    changed_data.loc[current_and_future, "hub_non_ev_power_kw"] += 20
    changed_data.loc[current_and_future, "ambient_temperature_c"] += 30
    after = study.make_supervised(changed_data, changed_circuits)
    issued = before.origin == issue
    assert issued.sum() == 80
    pd.testing.assert_frame_equal(before.x.loc[issued], after.x.loc[issued])
    assert np.all(after.y[issued] - before.y[issued] == 20)

    changed_circuits = circuits.copy()
    changed_circuits.loc[issue - study.INTERVAL, "water_heater_kw"] += 1
    observed_change = study.make_supervised(data, changed_circuits)
    np.testing.assert_allclose(
        observed_change.x.loc[issued, "water_heater_lag_1_kw"]
        - before.x.loc[issued, "water_heater_lag_1_kw"], 1,
    )
    pd.testing.assert_frame_equal(
        observed_change.x.loc[issued, list(before.feature_sets[study.METHODS[0]])],
        before.x.loc[issued, list(before.feature_sets[study.METHODS[0]])],
    )


def test_submeter_history_uses_completed_blocks_and_keeps_same_aggregate_target():
    data, circuits = observations()
    rows = study.make_supervised(data, circuits)
    origin_pos = 7 * 96 + 8
    issue = data.index[origin_pos]
    x = rows.x.loc[rows.origin == issue].iloc[0]
    for column, prefix in study.CIRCUIT_PREFIXES.items():
        for lag in study.POWER_LAGS:
            assert x[f"{prefix}_lag_{lag}_kw"] == circuits[column].iloc[origin_pos - lag]
        for steps in study.ROLLING_STEPS:
            history = circuits[column].iloc[origin_pos - steps:origin_pos]
            assert x[f"{prefix}_mean_prev_{steps}_kw"] == pytest.approx(history.mean())
            assert x[f"{prefix}_max_prev_{steps}_kw"] == history.max()
            assert x[f"{prefix}_count_prev_{steps}"] == steps
        assert x[f"{prefix}_slope_1h_kw"] == pytest.approx(
            circuits[column].iloc[origin_pos - 1] - circuits[column].iloc[origin_pos - 5]
        )
    np.testing.assert_array_equal(rows.y[rows.origin == issue],
                                  data["hub_non_ev_power_kw"].iloc[origin_pos:origin_pos + 80])


def test_twenty_hour_purge_and_identical_complete_validation_and_test_grids():
    data, circuits = observations(42)
    rows = study.make_supervised(data, circuits)
    cutoff = data.index[0] + 5 * study.WEEK
    end = data.index[0] + 6 * study.WEEK
    for weeks, expected_origins in ((2, 317), (3, 485), (4, 653)):
        fit = rows.mask_fit(cutoff - weeks * study.WEEK, cutoff)
        assert rows.origin[fit].nunique() == expected_origins
        assert fit.sum() == expected_origins * 80
        assert (rows.origin[fit] + 80 * study.INTERVAL <= cutoff).all()
        assert not fit[rows.origin == cutoff - pd.Timedelta(hours=19)].any()
    test = rows.mask_evaluation(cutoff, end)
    assert rows.origin[test].nunique() == 593
    assert test.sum() == 593 * 80
    assert rows.target[test].max() + study.INTERVAL == end
    assert rows.origin[test].max() == end - pd.Timedelta(hours=20)
    for offset in study.CV_DAY_OFFSETS:
        start = data.index[0] + 4 * study.WEEK + offset * study.DAY
        valid = rows.mask_evaluation(start, start + study.DAY)
        assert rows.origin[valid].nunique() == 17
        assert valid.sum() == 17 * 80
        assert rows.origin[valid].min() == start
        assert rows.origin[valid].max() == start + pd.Timedelta(hours=4)


def raw_example(tmp_path):
    index = pd.date_range("2018-01-07", periods=30, freq="min", name="Time")
    raw = pd.DataFrame({
        "Water Heating Electric Power (kW)": 0.4,
        "HVAC Heating Main Power (kW)": 1.5,
        "HVAC Cooling Main Power (kW)": 0.05,
        "HVAC Heating Fan Power (kW)": 0.2,
        "HVAC Cooling Fan Power (kW)": 0.0,
        "HVAC Heating ER Power (kW)": 3.0,
        "HVAC Heating Electric Power (kW)": 4.7,
        "HVAC Cooling Electric Power (kW)": 0.05,
        "DC House Hub Other Electric Power (kW)": 0.15,
    }, index=index)
    data = pd.DataFrame({"hub_non_ev_power_kw": [2.3, 2.3]},
                        index=pd.date_range(index[0], periods=2, freq="15min", name="timestamp"))
    path = tmp_path / "raw.csv"
    raw.to_csv(path)
    return raw, data, path


def test_raw_projection_preserves_circuit_sum_excludes_er_and_requires_every_minute(tmp_path):
    raw, data, path = raw_example(tmp_path)
    circuits, audit = study.read_submeter_power(path, data)
    np.testing.assert_allclose(circuits.outdoor_heat_pump_kw, 1.55)
    np.testing.assert_allclose(circuits.indoor_fan_kw, 0.2)
    np.testing.assert_allclose(circuits.water_heater_kw, 0.4)
    assert audit["raw_minutes"] == 30
    raw["HVAC Heating ER Power (kW)"] += 7
    raw["HVAC Heating Electric Power (kW)"] += 7
    raw.to_csv(path)
    after, _ = study.read_submeter_power(path, data)
    pd.testing.assert_frame_equal(circuits, after)
    raw.iloc[1:].to_csv(path)
    with pytest.raises(ValueError, match="every minute"):
        study.read_submeter_power(path, data)


def test_raw_projection_rejects_hub_target_that_includes_resistance(tmp_path):
    _raw, data, path = raw_example(tmp_path)
    data["hub_non_ev_power_kw"] += 3
    with pytest.raises(ValueError, match="do not equal corrected Hub"):
        study.read_submeter_power(path, data)
