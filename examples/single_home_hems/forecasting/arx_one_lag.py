"""Causal one-lag ARX forecast for 15-minute rest-of-home mean power.

The model is an ordinary least-squares fit of

    y[t] = beta[0] + beta[1] * y[t-1]
           + beta[2:4] * sin/cos(target clock time)
           + beta[4:6] * sin/cos(target weekday) + error[t].

Only originally observed target/lag pairs in the preceding 20 calendar days
are used in fitting. Forecasts are recursive and every forecast bin is clipped
to nonnegative watts before being used as the next lag.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


INTERVAL = pd.Timedelta(minutes=15)
TRAIN_DAYS = 20
HORIZON_STEPS = 48
FEATURE_COLUMNS = (
    "intercept", "lag_1_w", "target_clock_sin", "target_clock_cos",
    "target_week_sin", "target_week_cos",
)


@dataclass(frozen=True)
class ARXOneLagModel:
    coefficients: np.ndarray


def _calendar_design(index: pd.DatetimeIndex) -> np.ndarray:
    clock = 2 * np.pi * (index.hour * 60 + index.minute) / 1440
    week = 2 * np.pi * index.dayofweek / 7
    return np.column_stack((
        np.ones(len(index), dtype=float),
        np.sin(clock), np.cos(clock), np.sin(week), np.cos(week),
    ))


def fit_arx_one_lag(
    observed: pd.Series,
    train_end: pd.Timestamp,
    *,
    train_days: int = TRAIN_DAYS,
) -> tuple[ARXOneLagModel, dict]:
    """Fit once on observed target/previous-bin pairs in the training window."""
    train_end = pd.Timestamp(train_end)
    if train_days <= 0 or train_end != train_end.normalize():
        raise ValueError("Training requires a positive count of complete calendar days")
    train_start = train_end - pd.Timedelta(days=train_days)
    window = observed.loc[(observed.index >= train_start) & (observed.index < train_end)]
    expected = pd.date_range(train_start, train_end - INTERVAL, freq=INTERVAL)
    if not isinstance(window.index, pd.DatetimeIndex) or not window.index.equals(expected):
        raise ValueError("Training requires a regular 15-minute calendar grid")
    target = window.to_numpy(dtype=float)
    lag = window.shift(1).to_numpy(dtype=float)
    usable = np.isfinite(target) & np.isfinite(lag)
    if usable.sum() <= len(FEATURE_COLUMNS):
        raise ValueError("Insufficient originally observed target/lag pairs for ARX")
    calendar = _calendar_design(window.index)
    design = np.column_stack((calendar[:, 0], lag, calendar[:, 1:]))[usable]
    response = target[usable]
    coefficients, _, rank, _ = np.linalg.lstsq(design, response, rcond=None)
    if rank < len(FEATURE_COLUMNS):
        raise ValueError("ARX design matrix has insufficient rank")
    model = ARXOneLagModel(coefficients=coefficients)
    details = {
        "start_inclusive": train_start.isoformat(),
        "end_exclusive": train_end.isoformat(),
        "calendar_days": train_days,
        "observed_bins": int(np.isfinite(target).sum()),
        "training_rows": int(usable.sum()),
        "feature_columns": list(FEATURE_COLUMNS),
        "coefficients": dict(zip(FEATURE_COLUMNS, coefficients.tolist())),
        "fit_method": "ordinary least squares",
        "fit_schedule": "once before the evaluation period",
        "training_rule": "target and immediately previous 15-minute bin originally observed within the training window",
    }
    return model, details


def predict_arx_one_lag(
    model: ARXOneLagModel,
    observed: pd.Series,
    origin: pd.Timestamp,
    *,
    steps: int = HORIZON_STEPS,
) -> np.ndarray:
    """Issue a recursive forecast using only the completed bin before origin."""
    origin = pd.Timestamp(origin)
    if steps <= 0:
        raise ValueError("steps must be positive")
    latest_time = origin - INTERVAL
    if latest_time not in observed.index:
        raise ValueError("Latest completed 15-minute bin is absent from the grid")
    latest_w = float(observed.loc[latest_time])
    if not np.isfinite(latest_w):
        raise ValueError("Latest completed bin must be originally observed")
    targets = pd.date_range(origin, periods=steps, freq=INTERVAL)
    calendar = _calendar_design(targets)
    coefficients = np.asarray(model.coefficients, dtype=float)
    if coefficients.shape != (len(FEATURE_COLUMNS),) or not np.isfinite(coefficients).all():
        raise ValueError("ARX model must contain six finite coefficients")
    forecast = np.empty(steps, dtype=float)
    prior = latest_w
    for step in range(steps):
        predicted = coefficients[0] + coefficients[1] * prior + np.dot(
            coefficients[2:], calendar[step, 1:]
        )
        prior = max(float(predicted), 0.0)
        forecast[step] = prior
    return forecast
