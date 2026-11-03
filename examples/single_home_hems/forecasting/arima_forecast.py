"""Causal ARIMA forecasts of 15-minute rest-of-home power.

The model is fitted once on the preceding 20 calendar days. Missing original
observations remain NaN in the regular training series, so the state-space
likelihood handles them without synthetic power values. At validation time,
the fitted parameters stay fixed while the filter incorporates only bins that
have completed before each forecast origin.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


INTERVAL = pd.Timedelta(minutes=15)
TRAIN_DAYS = 20
HORIZON_STEPS = 48
ORDER = (1, 0, 1)
SUPPORTED_ORDERS = frozenset({ORDER, (2, 0, 2)})


def _validate_observed_index(observed: pd.Series) -> None:
    if not isinstance(observed, pd.Series) or not isinstance(observed.index, pd.DatetimeIndex):
        raise TypeError("observed must be a pandas Series with a DatetimeIndex")
    if observed.index.has_duplicates or not observed.index.is_monotonic_increasing:
        raise ValueError("observed timestamps must be unique and increasing")


def _validate_power_values(values: pd.Series) -> None:
    nonmissing = values.dropna().to_numpy(dtype=float)
    if not np.isfinite(nonmissing).all() or (nonmissing < 0).any():
        raise ValueError("Originally observed power must be finite and nonnegative")


class SequentialArimaForecaster:
    """Advance a frozen-parameter ARIMA filter through successive issue times."""

    def __init__(self, fitted_result, train_end: pd.Timestamp):
        self._result = fitted_result
        self.train_end = pd.Timestamp(train_end)
        self.last_ingested_bin = self.train_end - INTERVAL
        self.last_forecast_origin: pd.Timestamp | None = None
        self.fitted_parameters = fitted_result.params.copy()

    def forecast(
        self,
        observed: pd.Series,
        origin: pd.Timestamp,
        *,
        steps: int = HORIZON_STEPS,
    ) -> np.ndarray:
        """Issue a forecast using originally observed bins strictly before origin.

        A missing bin is passed to the Kalman filter as NaN, including when
        eligible origins are separated by a gap. Future observed values, even
        if present in ``observed``, are never read for this issue time.
        """
        origin = pd.Timestamp(origin)
        if steps <= 0:
            raise ValueError("steps must be positive")
        if origin < self.train_end or (origin - self.train_end) % INTERVAL:
            raise ValueError("origin must be on the 15-minute grid at or after training")
        if self.last_forecast_origin is not None and origin <= self.last_forecast_origin:
            raise ValueError("ARIMA forecast origins must be strictly increasing")
        _validate_observed_index(observed)
        if origin not in observed.index:
            raise ValueError("origin must be present on the source grid")

        last_available = origin - INTERVAL
        if last_available > self.last_ingested_bin:
            update_index = pd.date_range(
                self.last_ingested_bin + INTERVAL, last_available, freq=INTERVAL
            )
            newly_completed = observed.reindex(update_index).astype(float)
            _validate_power_values(newly_completed)
            self._result = self._result.extend(newly_completed)
            self.last_ingested_bin = last_available

        predicted = np.asarray(self._result.forecast(steps=steps), dtype=float)
        if len(predicted) != steps or not np.isfinite(predicted).all():
            raise ValueError("ARIMA produced a nonfinite or incomplete forecast")
        self.last_forecast_origin = origin
        return np.maximum(predicted, 0.0)


def fit_arima(
    observed: pd.Series,
    train_end: pd.Timestamp,
    *,
    train_days: int = TRAIN_DAYS,
    order: tuple[int, int, int] = ORDER,
    maxiter: int | None = None,
) -> tuple[SequentialArimaForecaster, dict]:
    """Fit an approved ARIMA order with constant on regular training days.

    The input must contain one 15-minute timestamp per bin in the requested
    window. Only originally observed power is supplied; missing bins are NaN.
    No validation or later-September observations enter parameter fitting.
    The original ARIMA(1,0,1) and its default optimizer settings remain the
    defaults; ARIMA(2,0,2) can use a larger ``maxiter`` to converge.
    """
    _validate_observed_index(observed)
    train_end = pd.Timestamp(train_end)
    if train_days <= 0:
        raise ValueError("train_days must be positive")
    if order not in SUPPORTED_ORDERS:
        raise ValueError("order must be (1, 0, 1) or (2, 0, 2)")
    if maxiter is not None and (not isinstance(maxiter, int) or maxiter <= 0):
        raise ValueError("maxiter must be a positive integer or None")
    train_start = train_end - pd.Timedelta(days=train_days)
    expected_index = pd.date_range(train_start, train_end - INTERVAL, freq=INTERVAL)
    window = observed.loc[(observed.index >= train_start) & (observed.index < train_end)]
    if not window.index.equals(expected_index):
        raise ValueError("ARIMA training requires the complete regular 15-minute grid")
    training = window.astype(float).reindex(expected_index)
    _validate_power_values(training)
    observed_bins = int(training.notna().sum())
    if observed_bins < 100:
        raise ValueError("Insufficient originally observed training bins for ARIMA")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        arima = ARIMA(training, order=order, trend="c")
        if maxiter is None:
            fitted = arima.fit()
        else:
            fitted = arima.fit(method_kwargs={"maxiter": maxiter})

    details = {
        "start_inclusive": train_start.isoformat(),
        "end_exclusive": train_end.isoformat(),
        "calendar_days": train_days,
        "training_bins": len(training),
        "observed_bins": observed_bins,
        "missing_bins": int(training.isna().sum()),
        "order": list(order),
        "fit_maxiter": maxiter,
        "trend": "c",
        "fit_schedule": "fit parameters once before the evaluation period",
        "state_update_rule": "extend frozen-parameter Kalman state through completed originally observed bins; missing bins stay NaN",
        "fit_converged": bool(fitted.mle_retvals.get("converged", False)),
        "aic": float(fitted.aic),
        "parameter_names": list(fitted.param_names),
        "parameters": {name: float(value) for name, value in fitted.params.items()},
        "fit_warnings": [str(item.message) for item in caught],
    }
    return SequentialArimaForecaster(fitted, train_end), details
