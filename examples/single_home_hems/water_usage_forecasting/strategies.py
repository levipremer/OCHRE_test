"""Small, interchangeable water-flow forecast baselines.

To add a strategy, implement ``fit`` and ``predict`` and register a no-argument
factory in ``STRATEGIES``. Fit only on training data. At prediction time, history
contains completed intervals; return one finite l/min value per future interval.
Indexes label interval starts and use the same local timezone throughout.
"""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np
import pandas as pd


class ForecastStrategy(Protocol):
    """Contract for a strategy evaluated by the comparison runner."""

    def fit(self, training: pd.Series) -> None:
        """Learn parameters from training observations, ignoring missing values."""
        ...

    def predict(
        self, history: pd.Series, future_index: pd.DatetimeIndex
    ) -> np.ndarray:
        """Forecast future interval means using only already completed history."""
        ...


def _finite_training(training: pd.Series) -> pd.Series:
    if not isinstance(training.index, pd.DatetimeIndex):
        raise TypeError("Training must have a DatetimeIndex")
    finite = training.loc[np.isfinite(training.to_numpy(dtype=float))]
    if finite.empty:
        raise ValueError("Training data must contain at least one finite value")
    return finite


def _slots(index: pd.DatetimeIndex, weekly: bool = False) -> np.ndarray:
    """Use local wall-clock minutes; both occurrences of a DST hour share a slot."""
    minutes = np.asarray(index.hour * 60 + index.minute)
    return minutes + np.asarray(index.dayofweek) * 1440 if weekly else minutes


class Zero:
    """Always forecast zero flow; useful for sparse water-use observations."""

    def fit(self, training: pd.Series) -> None:
        _finite_training(training)

    def predict(
        self, history: pd.Series, future_index: pd.DatetimeIndex
    ) -> np.ndarray:
        return np.zeros(len(future_index), dtype=float)


class _TrainingMean:
    def __init__(self) -> None:
        self.mean: float | None = None

    def fit(self, training: pd.Series) -> None:
        self.mean = float(_finite_training(training).mean())

    def _fallback(self) -> float:
        if self.mean is None:
            raise RuntimeError("Call fit before predict")
        return self.mean


class HistoricalMean(_TrainingMean):
    """Repeat the mean of the observations provided to the latest fit."""

    def predict(self, history: pd.Series, future_index: pd.DatetimeIndex) -> np.ndarray:
        return np.full(len(future_index), self._fallback(), dtype=float)


class Persistence(_TrainingMean):
    """Repeat the last finite completed observation, or the training mean."""

    def predict(
        self, history: pd.Series, future_index: pd.DatetimeIndex
    ) -> np.ndarray:
        fallback = self._fallback()
        if future_index.empty:
            return np.empty(0, dtype=float)
        completed = history.loc[history.index < future_index[0]].sort_index()
        values = completed.to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        value = float(finite[-1]) if finite.size else fallback
        return np.full(len(future_index), value, dtype=float)


class _Profile(_TrainingMean):
    weekly = False

    def __init__(self) -> None:
        super().__init__()
        self.profile: dict[int, float] = {}

    def fit(self, training: pd.Series) -> None:
        finite = _finite_training(training)
        self.mean = float(finite.mean())
        self.profile = finite.groupby(_slots(finite.index, self.weekly)).mean().to_dict()

    def predict(
        self, history: pd.Series, future_index: pd.DatetimeIndex
    ) -> np.ndarray:
        fallback = self._fallback()
        return np.asarray(
            [self.profile.get(int(slot), fallback) for slot in _slots(future_index, self.weekly)],
            dtype=float,
        )


class DailyProfile(_Profile):
    """Training mean for each local time-of-day slot; unseen slots use the mean."""


class WeeklyProfile(_Profile):
    """Training mean for each weekday and local time-of-day slot."""

    weekly = True


class _Seasonal(_Profile):
    days = 1

    def predict(
        self, history: pd.Series, future_index: pd.DatetimeIndex
    ) -> np.ndarray:
        fallback = super().predict(history, future_index)
        if future_index.empty:
            return fallback

        # Work in local calendar time so "yesterday at 08:00" stays 08:00 across
        # DST. For a repeated wall-clock hour, the latest completed value wins.
        # No localization is needed, so nonexistent spring-forward times simply
        # miss the lookup and receive the fitted profile fallback.
        offset = pd.DateOffset(days=self.days)
        analog_times = future_index.tz_localize(None) - offset
        local_history = history.index.tz_localize(None)
        needed = (
            (history.index < future_index[0])
            & (local_history >= analog_times.min())
            & (local_history <= analog_times.max())
        )
        completed = history.loc[needed].sort_index()
        lookup = {
            timestamp.tz_localize(None): float(value)
            for timestamp, value in completed.items()
        }
        predictions = np.empty(len(future_index), dtype=float)
        for i, timestamp in enumerate(future_index):
            local_time = timestamp.tz_localize(None)
            value = lookup.get(local_time - offset, np.nan)
            predictions[i] = value if np.isfinite(value) else fallback[i]
            # A horizon longer than one season repeats its own predictions,
            # never actual observations from within the forecast window.
            lookup[local_time] = predictions[i]
        return predictions


class PreviousDay(_Seasonal):
    """Repeat the previous local calendar day; missing values use DailyProfile."""


class PreviousWeek(_Seasonal):
    """Repeat the previous local calendar week; missing values use WeeklyProfile."""

    days = 7
    weekly = True


def _optional_strategy(module: str, class_name: str) -> Callable[[], ForecastStrategy]:
    """Keep learned-model dependencies optional for baseline-only runs."""
    def create() -> ForecastStrategy:
        from importlib import import_module
        return getattr(import_module(f".{module}", __package__), class_name)()
    return create


STRATEGIES: dict[str, Callable[[], ForecastStrategy]] = {
    "rf_prophet_hybrid": _optional_strategy("hybrid", "PaperHybrid"),
    "ets_additive_seasonal": _optional_strategy("statistical", "AdditiveSeasonalETS"),
    "sarima_daily": _optional_strategy("statistical", "DailySARIMA"),
    "sarima_weekly": _optional_strategy("statistical", "WeeklySARIMA"),
    "stl_seasonal_naive": _optional_strategy("statistical", "STLSeasonalNaive"),
    "stl_ets": _optional_strategy("statistical", "STLETS"),
    "stl_arima": _optional_strategy("statistical", "STLARIMA"),
    "feedforward_nn": _optional_strategy("neural", "FeedForwardNN"),
    "cumulative_feedforward_nn": _optional_strategy("neural", "CumulativeFeedForwardNN"),
    "historical_mean": HistoricalMean,
    "zero": Zero,
    "persistence": Persistence,
    "previous_day": PreviousDay,
    "previous_week": PreviousWeek,
    "daily_profile": DailyProfile,
    "weekly_profile": WeeklyProfile,
}

STRATEGY_LABELS = {
    "rf_prophet_hybrid": "Random forest / Prophet / persistence",
    "ets_additive_seasonal": "ETS (additive seasonal)",
    "sarima_daily": "SARIMA (daily)",
    "sarima_weekly": "SARIMA (weekly)",
    "stl_seasonal_naive": "STL + seasonal naive",
    "stl_ets": "STL + ETS",
    "stl_arima": "STL + ARIMA",
    "feedforward_nn": "Feedforward neural network",
    "cumulative_feedforward_nn": "Cumulative feedforward neural network",
    "historical_mean": "Historical mean",
    "zero": "Zero",
    "persistence": "Last-value persistence",
    "previous_day": "Previous day",
    "previous_week": "Previous week",
    "daily_profile": "Mean daily profile",
    "weekly_profile": "Mean weekly profile",
}
