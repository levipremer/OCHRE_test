"""Retrospective selection of a winter critical-peak pricing event day."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


LOCAL_TIMEZONE = "America/Indiana/Indianapolis"
MORNING_START_HOUR = 6
MORNING_END_HOUR = 8


@dataclass(frozen=True)
class CPPEventSelection:
    """One event selected using realized weather, with a simulated notice time."""

    event_date: dt.date
    event_start: pd.Timestamp
    event_end: pd.Timestamp
    notice_time: pd.Timestamp
    morning_mean_temperature_c: float
    eligible_morning_means_c: dict[dt.date, float]
    weather_file: Path
    selection_basis: str = "retrospective_realized_06_08_mean_temperature"


def select_retrospective_coldest_weekday(
    weather_file: str | Path,
    start_time: pd.Timestamp | dt.datetime,
    duration: pd.Timedelta | dt.timedelta,
    *,
    local_timezone: str = LOCAL_TIMEZONE,
    notice_hour: int = 16,
) -> CPPEventSelection:
    """Choose the weekday with the coldest realized 06:00–08:00 local mean.

    A candidate must have its complete two-hour window inside the simulation
    and all 120 one-minute weather observations. Ties go to the earlier date.
    This is an offline stress-scenario selection, not a causal utility forecast.
    """
    if not 0 <= notice_hour < 24:
        raise ValueError("notice_hour must be between 0 and 23")
    start = pd.Timestamp(start_time)
    if start.tzinfo is None:
        raise ValueError("start_time must be timezone-aware")
    start = start.tz_convert(local_timezone)
    duration = pd.Timedelta(duration)
    if pd.isna(duration) or duration <= pd.Timedelta(0):
        raise ValueError("duration must be positive")
    end = start + duration

    source = Path(weather_file).expanduser().resolve()
    weather = pd.read_csv(source, usecols=["Time", "temp_air"])
    time = pd.DatetimeIndex(pd.to_datetime(weather["Time"], utc=True)).tz_convert(local_timezone)
    temperature = pd.Series(pd.to_numeric(weather["temp_air"], errors="coerce").to_numpy(), index=time)

    means: dict[dt.date, float] = {}
    for date in pd.date_range(start.normalize(), end.normalize(), freq="D"):
        if date.weekday() >= 5:
            continue
        morning_start = date + pd.Timedelta(hours=MORNING_START_HOUR)
        morning_end = date + pd.Timedelta(hours=MORNING_END_HOUR)
        if morning_start < start or morning_end > end:
            continue
        observed = temperature.loc[(temperature.index >= morning_start) & (temperature.index < morning_end)]
        expected = pd.date_range(morning_start, periods=120, freq="min")
        if not observed.index.equals(expected) or not np.isfinite(observed.to_numpy(dtype=float)).all():
            raise ValueError(f"Incomplete 06:00–08:00 realized weather for {date.date()}")
        means[date.date()] = float(observed.mean())

    if not means:
        raise ValueError("Simulation contains no complete weekday 06:00–08:00 weather window")
    selected_date = min(means, key=lambda date: (means[date], date))
    event_start = pd.Timestamp(selected_date).tz_localize(local_timezone) + pd.Timedelta(hours=MORNING_START_HOUR)
    notice_date = selected_date - dt.timedelta(days=1)
    notice_time = pd.Timestamp(
        dt.datetime.combine(notice_date, dt.time(hour=notice_hour))
    ).tz_localize(local_timezone)
    return CPPEventSelection(
        event_date=selected_date,
        event_start=event_start,
        event_end=event_start + pd.Timedelta(hours=MORNING_END_HOUR - MORNING_START_HOUR),
        notice_time=notice_time,
        morning_mean_temperature_c=means[selected_date],
        eligible_morning_means_c=means,
        weather_file=source,
    )
