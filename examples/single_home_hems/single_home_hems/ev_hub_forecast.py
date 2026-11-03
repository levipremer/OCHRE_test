"""Issued 15-minute Home Hub forecasts for EV load-following control.

``eligible_*`` excludes any Hub circuits that cannot be backed by the EV,
while ``hub_mean_kw`` is the full non-EV Hub demand for whole-home accounting.
The capped mean is an explicit model output: it cannot in general be recovered
from the uncapped 15-minute mean and minimum.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


INTERVAL = pd.Timedelta(minutes=15)
MOBILITY_SPLIT_POLICIES = ("reject", "hold_parent_interval_coefficients")
REQUIRED_COLUMNS = (
    "eligible_load_mean_kw",  # mean minute demand before inverter clipping
    "eligible_mean_kw",       # mean(min(minute demand, inverter limit))
    "eligible_min_kw",        # min(minute demand, inverter limit)
    "hub_mean_kw",            # non-EV Home Hub mean; excludes resistance in EV-only wiring
)


@dataclass(frozen=True)
class EVHubIntervalForecast:
    """One causally issued set of complete 15-minute power predictions.

    Rows are labeled by local interval start. ``inputs_ready_at`` is the latest
    availability time of any observation used to issue these predictions.
    """

    issued_at: pd.Timestamp
    inputs_ready_at: pd.Timestamp
    intervals: pd.DataFrame
    source: str
    mobility_split_policy: str = "reject"

    def __post_init__(self):
        issued_at = pd.Timestamp(self.issued_at)
        inputs_ready_at = pd.Timestamp(self.inputs_ready_at)
        if pd.isna(issued_at) or pd.isna(inputs_ready_at) or inputs_ready_at > issued_at:
            raise ValueError("Home Hub forecast inputs must be ready no later than issuance")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("Home Hub forecast must identify its source")
        if self.mobility_split_policy not in MOBILITY_SPLIT_POLICIES:
            raise ValueError("Unknown Home Hub mobility split policy")
        if not isinstance(self.intervals, pd.DataFrame):
            raise TypeError("Home Hub forecast intervals must be a DataFrame")
        rows = self.intervals.copy()
        if not isinstance(rows.index, pd.DatetimeIndex) or rows.empty:
            raise ValueError("Home Hub forecast needs a nonempty DatetimeIndex")
        if rows.index.has_duplicates or not rows.index.is_monotonic_increasing:
            raise ValueError("Home Hub forecast interval starts must be unique and increasing")
        if any(t != t.floor(INTERVAL) for t in rows.index):
            raise ValueError("Home Hub forecast rows must start on 15-minute boundaries")
        missing = set(REQUIRED_COLUMNS) - set(rows)
        if missing:
            raise ValueError(f"Home Hub forecast lacks required columns: {sorted(missing)}")
        values = rows.loc[:, list(REQUIRED_COLUMNS)].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Home Hub forecast power must be finite and nonnegative")
        rows.loc[:, list(REQUIRED_COLUMNS)] = values
        raw, capped, minimum, hub = (rows[name].to_numpy(dtype=float) for name in REQUIRED_COLUMNS)
        if (minimum > capped + 1e-9).any() or (capped > raw + 1e-9).any() or (raw > hub + 1e-9).any():
            raise ValueError("Home Hub forecast means and minimum are physically inconsistent")
        object.__setattr__(self, "issued_at", issued_at)
        object.__setattr__(self, "inputs_ready_at", inputs_ready_at)
        object.__setattr__(self, "intervals", rows)

    def window(self, timestamp: pd.Timestamp, horizon: pd.Timedelta, inverter_kw: float) -> pd.DataFrame:
        """Return complete issued rows for the requested aligned horizon."""
        timestamp = pd.Timestamp(timestamp)
        horizon = pd.Timedelta(horizon)
        if timestamp != timestamp.floor(INTERVAL) or horizon <= pd.Timedelta(0) or horizon % INTERVAL:
            raise ValueError("Home Hub forecast request must use a positive 15-minute-aligned horizon")
        if self.issued_at > timestamp or self.inputs_ready_at > timestamp:
            raise ValueError("Home Hub forecast was not available at the EV decision time")
        if not np.isfinite(inverter_kw) or inverter_kw <= 0:
            raise ValueError("EV inverter limit must be positive and finite")
        expected = pd.date_range(timestamp, timestamp + horizon, freq=INTERVAL, inclusive="left")
        rows = self.intervals.reindex(expected)
        if rows.loc[:, list(REQUIRED_COLUMNS)].isna().any().any():
            raise ValueError("Home Hub forecast lacks complete 15-minute horizon coverage")
        if (rows["eligible_mean_kw"] > inverter_kw + 1e-9).any() or (
            rows["eligible_min_kw"] > inverter_kw + 1e-9
        ).any():
            raise ValueError("Home Hub eligible forecast exceeds the EV inverter limit")
        return rows
