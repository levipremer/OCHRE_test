"""Settled bill calculations for the single-home winter rate study."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .demand_windows import completed_demand_windows, demand_window_is_on_peak
from .rates import LOCAL_TIMEZONE


WINTER_DAILY_DEMAND_RATE_USD_PER_KW = 0.71


@dataclass(frozen=True)
class DailyDemandBill:
    """Daily peak details and their aggregate demand charge."""

    per_day: pd.DataFrame
    total_charge_usd: float

    @property
    def on_peak_charge_usd(self) -> float:
        return float(self.per_day["on_peak_demand_charge_usd"].sum())

    @property
    def anytime_charge_usd(self) -> float:
        return float(self.per_day["anytime_demand_charge_usd"].sum())


def winter_daily_demand_bill(
    whole_home_net_power_kw: pd.Series,
    *,
    rate_usd_per_kw_day: float = WINTER_DAILY_DEMAND_RATE_USD_PER_KW,
    anytime_rate_usd_per_kw_day: float = 0.0,
    window_step_minutes: int | None = None,
) -> DailyDemandBill:
    """Bill independent weekday on-peak and optional 24-hour maxima.

    With the Grid 2 anytime component enabled, complete 30-minute windows
    advance every 15 minutes and stay inside one local weekday. On-peak windows
    must be entirely inside one tariff period. The older on-peak-only tariff
    retains its aligned 30-minute blocks.
    An explicit ``window_step_minutes=30`` supports auditing historical bills.
    Its demand is the mean signed whole-home meter power, floored at zero to
    obtain net import. The 06:00-08:00 and 17:00-21:00 windows share one daily
    maximum. Only on-peak blocks present in the supplied simulation period are
    billed, so a partial first or last day is identified by ``interval_count``.
    The optional anytime term includes on-peak hours, but has its own maximum.
    Both components exclude weekends. Charges are settled dollars with no
    optimizer time-step/horizon normalization. Internal minute gaps are rejected.
    Rolling windows require 30 readings,
    so partial edge windows are omitted; legacy partial blocks are rejected.
    """
    if not isinstance(whole_home_net_power_kw, pd.Series):
        raise TypeError("Whole-home net power must be a pandas Series")
    index = whole_home_net_power_kw.index
    if not isinstance(index, pd.DatetimeIndex) or index.empty:
        raise ValueError("Whole-home power needs a nonempty DatetimeIndex")
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("Whole-home power timestamps must be unique and increasing")
    if (index.asi8 % pd.Timedelta(minutes=1).value != 0).any():
        raise ValueError("Whole-home power timestamps must align with minute starts")
    if len(index) > 1 and not (index[1:] - index[:-1] == pd.Timedelta(minutes=1)).all():
        raise ValueError("Whole-home power must contain consecutive one-minute readings")
    power = whole_home_net_power_kw.astype(float)
    if not np.isfinite(power.to_numpy()).all():
        raise ValueError("Whole-home net power must be finite")
    for name, rate in (("Daily demand rate", rate_usd_per_kw_day),
                       ("Anytime daily demand rate", anytime_rate_usd_per_kw_day)):
        if isinstance(rate, bool) or not np.isfinite(rate) or rate < 0:
            raise ValueError(f"{name} must be finite and nonnegative")

    if window_step_minutes is None:
        window_step_minutes = 15 if anytime_rate_usd_per_kw_day > 0 else 30
    if window_step_minutes not in (15, 30):
        raise ValueError("Demand window step must be 15 or 30 minutes")
    if window_step_minutes == 15:
        if power.index.tz is not None:
            power = power.tz_convert(LOCAL_TIMEZONE)
        means = completed_demand_windows(power)
        on_peak = np.array([demand_window_is_on_peak(t) for t in means.index], dtype=bool)
        import_kw = means.loc[on_peak].clip(lower=0)
        anytime_kw = means.clip(lower=0) if anytime_rate_usd_per_kw_day > 0 else means.iloc[:0]
    else:
        blocks = power.resample("30min", closed="left", label="left")
        means, counts = blocks.mean(), blocks.count()
        starts = means.index
        on_peak = (
            (starts.dayofweek < 5)
            & (((starts.hour >= 6) & (starts.hour < 8))
               | ((starts.hour >= 17) & (starts.hour < 21)))
        )
        if counts.loc[on_peak].ne(30).any():
            raise ValueError("Every observed on-peak demand block must have 30 one-minute readings")
        import_kw = means.loc[on_peak].clip(lower=0)
        anytime = (starts.dayofweek < 5) & (anytime_rate_usd_per_kw_day > 0)
        if counts.loc[anytime].ne(30).any():
            raise ValueError("Every observed anytime demand block must have 30 one-minute readings")
        anytime_kw = means.loc[anytime].clip(lower=0)
    days = import_kw.index.normalize().union(anytime_kw.index.normalize()).unique().sort_values()
    rows = []
    for local_day in days:
        daily = import_kw.loc[import_kw.index.normalize() == local_day]
        daily_anytime = anytime_kw.loc[anytime_kw.index.normalize() == local_day]
        peak_start = daily.idxmax() if len(daily) else pd.NaT
        peak_kw = float(daily.max()) if len(daily) else 0.0
        anytime_peak_start = daily_anytime.idxmax() if len(daily_anytime) else pd.NaT
        anytime_peak_kw = float(daily_anytime.max()) if len(daily_anytime) else 0.0
        on_peak_charge = peak_kw * float(rate_usd_per_kw_day)
        anytime_charge = anytime_peak_kw * float(anytime_rate_usd_per_kw_day)
        rows.append({
            "local_day": local_day,
            "peak_kw": peak_kw,
            "peak_interval_start": peak_start,
            "interval_count": len(daily),
            "on_peak_demand_charge_usd": on_peak_charge,
            "anytime_peak_kw": anytime_peak_kw,
            "anytime_peak_interval_start": anytime_peak_start,
            "anytime_interval_count": len(daily_anytime),
            "anytime_demand_charge_usd": anytime_charge,
            "demand_charge_usd": on_peak_charge + anytime_charge,
        })
    per_day = pd.DataFrame.from_records(rows, columns=(
        "local_day", "peak_kw", "peak_interval_start", "interval_count",
        "on_peak_demand_charge_usd", "anytime_peak_kw", "anytime_peak_interval_start",
        "anytime_interval_count", "anytime_demand_charge_usd", "demand_charge_usd",
    )).set_index("local_day")
    return DailyDemandBill(per_day, float(per_day["demand_charge_usd"].sum()))
