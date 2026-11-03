"""Causal, matching-clock fixed-load tails from a complete earlier local day."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .joint_study_forecasts import FIXED_TARGETS


INTERVAL = pd.Timedelta(minutes=15)
MINUTE = pd.Timedelta(minutes=1)
RF_STEPS = 32
EXTENDED_STEPS = 80


def prior_complete_local_day(
    minute_start: pd.DataFrame, timestamp: pd.Timestamp,
) -> tuple[pd.Timestamp, pd.DataFrame]:
    """Return the newest complete earlier 00:00--24:00 local meter day.

    The caller supplies minute-start power values derived only from readings
    completed by ``timestamp``. A clock-change day is skipped because it does
    not have one unambiguous reading for every local clock minute.
    """
    timestamp = pd.Timestamp(timestamp)
    if not isinstance(minute_start, pd.DataFrame) or minute_start.empty:
        raise ValueError("Prior-day load forecast needs completed minute meters")
    index = minute_start.index
    if not isinstance(index, pd.DatetimeIndex) or index.hasnans or not index.is_monotonic_increasing:
        raise ValueError("Prior-day meter history needs ordered minute timestamps")
    if index.tz is not None:
        if timestamp.tz is None:
            raise ValueError("Aware meter history requires an aware forecast issue")
        index = index.tz_convert(timestamp.tz).tz_localize(None)
    elif timestamp.tz is not None:
        # Existing RF providers use local-naive meter timestamps internally.
        index = pd.DatetimeIndex(index)
    wall = timestamp.tz_localize(None) if timestamp.tz is not None else timestamp
    if wall != wall.floor(INTERVAL):
        raise ValueError("Prior-day forecast issue must align to 15 minutes")
    local = minute_start.copy()
    local.index = index
    if (index.asi8 % MINUTE.value).any():
        raise ValueError("Prior-day meter timestamps must mark minute starts")
    local = local.loc[local.index < wall.normalize()]
    for day in sorted(local.index.normalize().unique(), reverse=True):
        if timestamp.tz is not None:
            # A provider may deduplicate repeated fall-back wall minutes while
            # preparing RF features. Check the calendar day itself so such a
            # 25-hour day can never masquerade as a complete 24-hour analog.
            midnight = pd.Timestamp(day).tz_localize(timestamp.tz)
            next_midnight = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize(timestamp.tz)
            if next_midnight - midnight != pd.Timedelta(days=1):
                continue
        expected = pd.date_range(day, periods=24 * 60, freq=MINUTE)
        rows = local.loc[(local.index >= day) & (local.index < day + pd.Timedelta(days=1))]
        if (rows.index.equals(expected)
                and np.isfinite(rows.to_numpy(dtype=float)).all()
                and (rows.to_numpy(dtype=float) >= -1e-7).all()):
            return pd.Timestamp(day), rows
    raise ValueError("No complete unambiguous prior local meter day is available")


def clock_slot_forecast(
    minute_day: pd.DataFrame, targets: pd.DatetimeIndex,
    *, column: str,
) -> np.ndarray:
    """Map each future 15-minute clock slot to its prior-day mean."""
    if column not in minute_day or len(minute_day) != 24 * 60:
        raise ValueError("Prior-day clock profile needs a full day of minute power")
    values = minute_day[column].to_numpy(dtype=float).reshape(96, 15).mean(axis=1)
    slots = targets.hour * 4 + targets.minute // 15
    return values[np.asarray(slots, dtype=int)]


@dataclass
class PriorDayFixedLoadTail:
    """Keep a frozen eight-hour joint RF issue, then use prior-day fixed load.

    ``split`` may contain the complete offline replay, but every issuance uses
    only a fully completed local day before its issue timestamp. The target is
    one fixed component and excludes controlled HVAC, HPWH, and resistance.
    """

    base: object
    split: pd.DataFrame
    target: str

    def __post_init__(self) -> None:
        if self.target not in FIXED_TARGETS or self.target not in self.split:
            raise ValueError("Prior-day joint tail needs a fixed split target")
        self.minute = self.split.loc[:, [self.target]].copy()
        self.method_label = f"{self.base.method_label}__prior_complete_local_day_after_8h"
        self.metadata = dict(
            self.base.metadata,
            method_label=self.method_label,
            tail_policy="prior_complete_local_day",
            tail_first_step=RF_STEPS,
            target=self.target,
        )
        self.history: list[dict[str, object]] = []

    def forecast_kw(self, timestamp: pd.Timestamp, steps: int = RF_STEPS) -> pd.Series:
        if isinstance(steps, bool) or not isinstance(steps, int) or steps not in (RF_STEPS, EXTENDED_STEPS):
            raise ValueError("Joint prior-day tail supports 32 or 80 forecast steps")
        timestamp = pd.Timestamp(timestamp)
        first = self.base.forecast_kw(timestamp, steps=RF_STEPS)
        if steps == RF_STEPS:
            self.history.append(dict(timestamp=str(timestamp), steps=steps, tail_source_day=None))
            return first
        source_day, minute_day = prior_complete_local_day(self.minute, timestamp)
        target_index = pd.date_range(
            timestamp + RF_STEPS * INTERVAL,
            periods=EXTENDED_STEPS - RF_STEPS,
            freq=INTERVAL,
        )
        tail = pd.Series(
            clock_slot_forecast(minute_day, target_index, column=self.target),
            index=target_index,
            name=first.name,
        )
        result = pd.concat((first, tail))
        expected = pd.date_range(timestamp, periods=steps, freq=INTERVAL)
        if not result.index.equals(expected) or not np.isfinite(result.to_numpy(dtype=float)).all():
            raise ValueError("Prior-day joint fixed-load forecast lacks complete 20-hour coverage")
        result.attrs.update(first.attrs)
        result.attrs.update(tail_policy="prior_complete_local_day",
                            tail_source_day=source_day.isoformat(), tail_first_step=RF_STEPS)
        self.history.append(dict(timestamp=str(timestamp), steps=steps,
                                 tail_source_day=source_day.isoformat()))
        return result
