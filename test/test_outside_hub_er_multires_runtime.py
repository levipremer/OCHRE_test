"""The deployed Outside-Hub RF matches study features and uses causal meters."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from examples.single_home_hems.single_home_hems.home_hub_observation import (
    HUB_POWER_KEY, TOTAL_POWER_KEY, project_ev_only_hub_observation,
)
from examples.single_home_hems.single_home_hems.outside_hub_er_multires_rf import (
    METHOD, STUDY_METADATA, MultiresWeatherOutsideHubRF,
)


class _FeatureChecksumModel:
    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return np.maximum(1.0 + matrix.mean(axis=1), 0.0)


def _columns() -> tuple[str, ...]:
    metadata = json.loads(STUDY_METADATA.read_text(encoding="utf-8"))
    return tuple(metadata["feature_columns"][METHOD])


def test_issue_ignores_future_plant_and_weather_values() -> None:
    issue = pd.Timestamp("2025-12-01 00:00:00-05:00")
    wall = issue.tz_localize(None)
    seed_index = pd.date_range(wall - pd.Timedelta(hours=24) + pd.Timedelta(minutes=1),
                               wall, freq="min", name="completed_at")
    seed = pd.Series(np.linspace(1.0, 2.0, len(seed_index)), index=seed_index)
    weather_index = pd.date_range(wall - pd.Timedelta(hours=24),
                                  wall + pd.Timedelta(hours=2), freq="min")
    weather = pd.DataFrame({
        "ambient_temperature_c": np.linspace(-8.0, -5.0, len(weather_index)),
        "ambient_relative_humidity": np.linspace(0.7, 0.8, len(weather_index)),
    }, index=weather_index)
    later_weather = weather.copy()
    later_weather.loc[later_weather.index >= wall, "ambient_temperature_c"] = 40.0
    later_weather.loc[later_weather.index >= wall, "ambient_relative_humidity"] = 0.1
    provider = MultiresWeatherOutsideHubRF(
        _FeatureChecksumModel(), _columns(), {}, seed, weather, issue.tz,
    )
    altered = MultiresWeatherOutsideHubRF(
        _FeatureChecksumModel(), _columns(), {}, seed, later_weather, issue.tz,
    )
    live_index = pd.date_range(issue - pd.Timedelta(hours=16) + pd.Timedelta(minutes=1),
                               issue + pd.Timedelta(minutes=15), freq="min")
    wall_index = live_index.tz_localize(None)
    power = seed.reindex(wall_index).fillna(3.0).to_numpy(float)
    observed = pd.DataFrame({HUB_POWER_KEY: 2.0, TOTAL_POWER_KEY: 2.0 + power},
                            index=live_index)
    changed = observed.copy()
    changed.loc[changed.index > issue, TOTAL_POWER_KEY] = 10_000.0
    original_forecast = provider.forecast_from_completed(issue, observed)
    changed_forecast = altered.forecast_from_completed(issue, changed)
    np.testing.assert_array_equal(original_forecast.to_numpy(), changed_forecast.to_numpy())
    expected_weather = provider._features(wall, seed)
    for feature, attribute in (
        ("future_ambient_temperature_c", "persisted_ambient_temperature_c"),
        ("future_ambient_relative_humidity", "persisted_ambient_relative_humidity"),
    ):
        assert original_forecast.attrs[attribute] == expected_weather.iloc[0][feature]
        assert original_forecast.attrs[attribute] == changed_forecast.attrs[attribute]
    assert original_forecast.attrs["weather_inputs_ready_at"] == issue
    assert original_forecast.attrs["persisted_weather_window_start"] == issue - pd.Timedelta(hours=1)
    assert original_forecast.attrs["persisted_weather_window_end_exclusive"] == issue
    assert provider.history[0]["latest_completed_meter_at"] == issue
    assert altered.history[0]["latest_completed_meter_at"] == issue


def test_total_home_meter_is_opt_in_and_er_submeter_stays_hidden() -> None:
    status = {
        "HVAC Heating Main Power (kW)": 1.0,
        "HVAC Heating Fan Power (kW)": 0.2,
        "HVAC Heating ER Power (kW)": 5.0,
        "HVAC Cooling Electric Power (kW)": 0.0,
        "Water Heating Electric Power (kW)": 0.4,
        "DC House Hub Other Electric Power (kW)": 0.5,
        "EV_01A Electric Power (kW)": 0.0,
        "EV_01A SOC (-)": 0.5,
        "EV_01A End SOC (-)": 0.5,
        "EV_01A SOH (-)": 1.0,
        "EV_01A Capacity (kWh)": 60.0,
        "EV_01A Parked": True,
        TOTAL_POWER_KEY: 7.1,
    }
    legacy = project_ev_only_hub_observation(status, "EV_01A")
    new = project_ev_only_hub_observation(status, "EV_01A", include_total_home=True)
    assert TOTAL_POWER_KEY not in legacy
    assert TOTAL_POWER_KEY in new
    assert "HVAC Heating ER Power (kW)" not in new
    assert new[TOTAL_POWER_KEY] - new[HUB_POWER_KEY] == 5.0
