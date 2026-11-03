"""Adapter for pre-issued, causal Home Hub forecasts from an external model."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .ev_hub_forecast import EVHubIntervalForecast, REQUIRED_COLUMNS


class IssuedEVHubForecasts:
    """Read complete 15-minute forecast snapshots; never select a future issue.

    One CSV row describes one forecast interval. Required columns are
    ``issued_at``, ``inputs_ready_at``, ``interval_start``, ``source``, and the
    four kW columns in :data:`REQUIRED_COLUMNS`. The adapter never estimates
    or fills a missing prediction.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        rows = pd.read_csv(self.path)
        required = {"issued_at", "inputs_ready_at", "interval_start", "source", *REQUIRED_COLUMNS}
        missing = required - set(rows)
        if missing or rows.empty:
            raise ValueError(f"Issued Home Hub CSV is empty or lacks columns: {sorted(missing)}")
        for name in ("issued_at", "inputs_ready_at", "interval_start"):
            rows[name] = pd.to_datetime(rows[name], errors="raise")
        issues: list[EVHubIntervalForecast] = []
        for issued_at, group in rows.groupby("issued_at", sort=True):
            ready = group["inputs_ready_at"].unique()
            sources = group["source"].unique()
            if len(ready) != 1 or len(sources) != 1:
                raise ValueError("Each Home Hub issuance must have one readiness time and source")
            if pd.isna(sources[0]) or not str(sources[0]).strip():
                raise ValueError("Each Home Hub issuance must identify a nonempty source")
            intervals = group.set_index("interval_start").loc[:, list(REQUIRED_COLUMNS)]
            issues.append(EVHubIntervalForecast(
                issued_at=issued_at,
                inputs_ready_at=pd.Timestamp(ready[0]),
                intervals=intervals,
                source=str(sources[0]),
            ))
        self.issues = tuple(issues)
        self.history: list[dict[str, str]] = []

    def issue(
        self,
        timestamp: pd.Timestamp,
        completed_history: pd.DataFrame,
        horizon: pd.Timedelta,
        inverter_kw: float,
    ) -> EVHubIntervalForecast:
        """Choose the newest known snapshot with full requested coverage."""
        timestamp = pd.Timestamp(timestamp)
        if not isinstance(completed_history, pd.DataFrame):
            raise TypeError("Home Hub observation history must be a DataFrame")
        if len(completed_history) and completed_history.index.max() > timestamp:
            raise ValueError("Home Hub forecast history contains an uncompleted plant interval")
        known = [item for item in self.issues if item.issued_at <= timestamp]
        if not known:
            raise ValueError(f"No issued Home Hub forecast is available at {timestamp}")
        chosen = known[-1]
        chosen.window(timestamp, horizon, inverter_kw)
        self.history.append({
            "decision_at": timestamp.isoformat(),
            "issued_at": chosen.issued_at.isoformat(),
            "inputs_ready_at": chosen.inputs_ready_at.isoformat(),
            "source": chosen.source,
        })
        return chosen
