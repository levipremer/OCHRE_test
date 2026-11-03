"""Past and future weather features for the EV-only Outside-Hub-plus-ER study.

At an issue time ``t``, past hour 1 is the mean of minute readings in
``[t - 1 hour, t)``; past hour 24 is ``[t - 24 hours, t - 23 hours)``.
Each mean requires every minute in its window. The causal future-weather
variant holds the latest complete past-hour mean over all forecast leads.
The oracle variant uses realized weather from each target's 15-minute interval
and must be labeled as a hindsight upper bound when results are reported.

The source is the minute weather exported by the OCHRE no-control baseline.
Neither variant uses an HVAC or resistance submeter.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


WEATHER_VARIANTS = ("past_weather_persistence", "past_weather_oracle")
MINUTE = pd.Timedelta(minutes=1)
TARGET_INTERVAL = pd.Timedelta(minutes=15)
HOUR = pd.Timedelta(hours=1)
PAST_HOURS = 24
WEATHER_SOURCES = (
    ("ambient_temperature_c", "ambient_temperature", "c"),
    ("ambient_relative_humidity", "ambient_relative_humidity", ""),
)
FUTURE_COLUMNS = (
    "future_ambient_temperature_c",
    "future_ambient_temperature_available",
    "future_ambient_relative_humidity",
    "future_ambient_relative_humidity_available",
)


def _read_minute_weather(path: Path) -> pd.DataFrame:
    weather = pd.read_csv(
        path,
        usecols=["timestamp", "ambient_temperature_c", "ambient_relative_humidity"],
        parse_dates=["timestamp"],
    ).set_index("timestamp")
    if weather.empty or weather.index.hasnans or weather.index.has_duplicates:
        raise ValueError("Minute weather has an empty, missing, or duplicate timestamp")
    expected = pd.date_range(weather.index[0], weather.index[-1], freq="min")
    if not weather.index.equals(expected):
        raise ValueError("Minute weather must have a sorted, complete one-minute clock")
    if weather.index[0].second != 0 or weather.index[0].microsecond != 0:
        raise ValueError("Minute weather timestamps must be aligned to whole minutes")
    weather = weather.apply(pd.to_numeric, errors="raise")
    if np.isinf(weather.to_numpy(dtype=float)).any():
        raise ValueError("Minute weather contains infinite values")
    humidity = weather["ambient_relative_humidity"].to_numpy(dtype=float)
    if np.any(np.isfinite(humidity) & ((humidity < 0.0) | (humidity > 1.0))):
        raise ValueError("Ambient relative humidity must be a fraction in [0, 1]")
    return weather


def _checked_pairs(
    origins: pd.DatetimeIndex,
    targets: pd.DatetimeIndex,
    weather_index: pd.DatetimeIndex,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    origins = pd.DatetimeIndex(origins)
    targets = pd.DatetimeIndex(targets)
    if len(origins) != len(targets) or origins.hasnans or targets.hasnans:
        raise ValueError("Issue and target times must be nonmissing paired timestamps")
    if origins.tz != weather_index.tz or targets.tz != weather_index.tz:
        raise ValueError("Issue, target, and weather timestamps must have the same timezone")
    if len(origins) and (
        (origins < weather_index[0]).any()
        or (origins > weather_index[-1] + MINUTE).any()
        or (targets < origins).any()
        or (targets < weather_index[0]).any()
        or (targets + TARGET_INTERVAL > weather_index[-1] + MINUTE).any()
    ):
        raise ValueError("Issue/target times fall outside the available 15-minute weather intervals")
    if len(origins) and (
        (origins.minute % 15 != 0).any()
        or (targets.minute % 15 != 0).any()
        or (origins.second != 0).any()
        or (targets.second != 0).any()
        or (origins.microsecond != 0).any()
        or (targets.microsecond != 0).any()
    ):
        raise ValueError("Issue and target times must align to the 15-minute forecast grid")
    return origins, targets


def _values_and_availability(
    source: pd.Series,
    index: pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray]:
    values = source.reindex(index).to_numpy(dtype=float)
    available = np.isfinite(values)
    return np.where(available, values, 0.0), available.astype(np.int8)


def weather_feature_matrices(
    origins: pd.DatetimeIndex,
    targets: pd.DatetimeIndex,
    weather_csv: Path,
) -> dict[str, pd.DataFrame]:
    """Return row-aligned causal-persistence and oracle weather matrices.

    The two matrices have identical columns and a RangeIndex matching the
    input issue-target pair order. Unavailable values are zero with an explicit
    availability flag. Every past value uses readings strictly before issue
    time; oracle future values alone use readings at or after the target time.
    """
    weather = _read_minute_weather(Path(weather_csv))
    origins, targets = _checked_pairs(origins, targets, weather.index)
    past: dict[str, np.ndarray] = {}
    persisted: dict[str, np.ndarray] = {}
    oracle: dict[str, np.ndarray] = {}
    unique_origins = origins.unique()

    for source_name, prefix, unit in WEATHER_SOURCES:
        source = weather[source_name]
        # At timestamp t, this is the mean of [t-60 minutes, t). Shifting the
        # rolling series by one minute excludes the current, unobserved minute.
        past_hour_mean = source.rolling(60, min_periods=60).mean().shift(1)
        for hour in range(1, PAST_HOURS + 1):
            lookback = unique_origins - (hour - 1) * HOUR
            value, available = _values_and_availability(past_hour_mean, lookback)
            unique_values = pd.Series(value, index=unique_origins)
            unique_flags = pd.Series(available, index=unique_origins)
            stem = f"{prefix}_prev_{hour}h"
            value_column = f"{stem}_{unit}" if unit else stem
            past[value_column] = unique_values.reindex(origins).to_numpy(dtype=float)
            past[f"{stem}_available"] = unique_flags.reindex(origins).to_numpy(dtype=np.int8)
            if hour == 1:
                future_stem = f"future_{prefix}"
                future_value_column = (
                    f"{future_stem}_{unit}" if unit else future_stem
                )
                persisted[future_value_column] = past[value_column].copy()
                persisted[f"{future_stem}_available"] = past[
                    f"{stem}_available"
                ].copy()

        # A target at s represents [s, s+15 minutes), including minute s+14.
        target_mean = source.rolling(15, min_periods=15).mean()
        value, available = _values_and_availability(
            target_mean, targets + TARGET_INTERVAL - MINUTE,
        )
        future_stem = f"future_{prefix}"
        future_value_column = f"{future_stem}_{unit}" if unit else future_stem
        oracle[future_value_column] = value
        oracle[f"{future_stem}_available"] = available

    common = pd.DataFrame(past, index=pd.RangeIndex(len(origins)))
    persistence = common.assign(**persisted)
    upper_bound = common.assign(**oracle)
    if tuple(persistence.columns[-4:]) != FUTURE_COLUMNS:
        raise AssertionError("Future weather feature schema changed unexpectedly")
    if not persistence.columns.equals(upper_bound.columns):
        raise AssertionError("Weather variants must share an identical feature schema")
    if not (np.isfinite(persistence.to_numpy(dtype=float)).all()
            and np.isfinite(upper_bound.to_numpy(dtype=float)).all()):
        raise ValueError("Weather features contain nonfinite values")
    return dict(zip(WEATHER_VARIANTS, (persistence, upper_bound), strict=True))
