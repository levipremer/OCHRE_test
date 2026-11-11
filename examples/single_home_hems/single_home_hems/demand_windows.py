"""Grid 2 rolling demand windows shared by forecasts, meters and billing."""

import numpy as np
import pandas as pd

from .rates import LOCAL_TIMEZONE, WinterTOURate


def demand_window_is_weekday(start):
    """A full 30-minute window must belong to one local weekday."""
    start = pd.Timestamp(start)
    if start.tz is not None:
        start = start.tz_convert(LOCAL_TIMEZONE)
    last = start + pd.Timedelta(minutes=29)
    return start.weekday() < 5 and start.date() == last.date()


def demand_window_is_on_peak(start):
    """Both quarters must be inside the same weekday on-peak period."""
    start = pd.Timestamp(start)
    return (demand_window_is_weekday(start)
            and WinterTOURate.is_on_peak(start)
            and WinterTOURate.is_on_peak(start + pd.Timedelta(minutes=29)))


def completed_demand_windows(power):
    """Signed means of 30 consecutive minute-start readings, every 15 minutes.

    Missing or partial windows are omitted. The caller supplies only readings
    available by its issue time (or within its billing evaluation window).
    """
    if power.empty:
        return power.copy()
    index = power.index
    local = index if index.tz is None else index.tz_convert(LOCAL_TIMEZONE)
    starts = index - pd.Timedelta(minutes=29)
    means = power.rolling(30, min_periods=30).mean()
    spans = index.to_series().diff(29)
    valid = (local.minute % 15 == 14) & (spans.to_numpy() == pd.Timedelta(minutes=29))
    valid &= np.array([demand_window_is_weekday(t) for t in starts])
    valid &= means.notna().to_numpy()
    return pd.Series(means.to_numpy()[valid], index=starts[valid], dtype=float)
