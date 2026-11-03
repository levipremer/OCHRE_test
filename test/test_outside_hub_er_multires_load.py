"""Timestamp and missingness checks for EV-only multiresolution load history."""

from __future__ import annotations

import numpy as np
import pandas as pd

from examples.single_home_hems.forecasting.evaluate_outside_hub_with_er import (
    load_composite_target,
)
from examples.single_home_hems.forecasting.outside_hub_er_multires_load import (
    load_feature_matrix,
    load_minute_observed,
)


def test_reconstructed_minute_meter_matches_original_target_bins() -> None:
    truth, observed, _ = load_composite_target()
    minute, audit = load_minute_observed(truth, observed)
    assert minute.index[0] == truth.index[0] + pd.Timedelta(minutes=1)
    assert minute.index[-1] == truth.index[-1] + pd.Timedelta(minutes=15)
    assert audit["observed_15min_bins"] == int(observed.notna().sum())
    assert audit["max_observed_bin_error_kw"] < 1e-10
    assert audit["max_truth_bin_error_kw"] < 1e-10
    assert audit["unavailable_minutes"] > 0


def test_windows_partition_past_24_hours_and_never_use_future_power() -> None:
    start = pd.Timestamp("2018-01-07 00:01:00")
    timestamps = pd.date_range(start, periods=1500, freq="min")
    values = np.arange(1500, dtype=float) / 100.0
    meter = pd.Series(values, index=timestamps)
    origin = timestamps[1440]
    # A post-issue power spike must have no effect on any feature.
    meter.loc[origin + pd.Timedelta(minutes=1)] = 100_000.0
    row = load_feature_matrix(meter, pd.DatetimeIndex([origin])).iloc[0]
    assert len(row) == 88
    np.testing.assert_allclose(row["load_minute_lag_0_kw"], values[1440])
    np.testing.assert_allclose(row["load_minute_lag_14_kw"], values[1426])
    np.testing.assert_allclose(
        row["load_prev_15m_block_1_kw"], values[1411:1426].mean(),
    )
    np.testing.assert_allclose(
        row["load_prev_15m_block_7_kw"], values[1321:1336].mean(),
    )
    np.testing.assert_allclose(
        row["load_prev_1h_block_1_kw"], values[1261:1321].mean(),
    )
    np.testing.assert_allclose(
        row["load_prev_1h_block_22_kw"], values[1:61].mean(),
    )
    assert row.filter(like="_available").eq(1).all()

    # One missing original minute invalidates its complete block only.
    meter.iloc[1414] = np.nan
    missing = load_feature_matrix(meter, pd.DatetimeIndex([origin])).iloc[0]
    assert missing["load_prev_15m_block_1_available"] == 0
    assert missing["load_prev_15m_block_1_kw"] == 0
    assert missing["load_minute_lag_14_available"] == 1
    assert missing["load_prev_1h_block_1_available"] == 1
