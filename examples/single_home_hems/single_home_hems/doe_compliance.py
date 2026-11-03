"""Realized Scenario 3 grid-envelope and panel-loading diagnostics.

These are power measurements, not a PCS current-protection certification. The
panel quantity is the study's gross-load proxy: net home power plus any V2H
discharge. It assumes the EV is the home's only local power source.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .rates import LOCAL_TIMEZONE


@dataclass(frozen=True)
class Scenario3DOECompliance:
    """Aligned 15-minute measurements and compact realized-violation metrics."""

    intervals: pd.DataFrame
    summary: dict[str, float | int | None]


def _peak_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """Original Scenario 3 weekday utility windows in West Lafayette local time."""
    local = index if index.tz is None else index.tz_convert(LOCAL_TIMEZONE)
    return np.asarray(
        (local.weekday < 5)
        & (((local.hour >= 6) & (local.hour < 9))
           | ((local.hour >= 17) & (local.hour < 21))),
        dtype=bool,
    )


def evaluate_scenario3_doe(
    data: pd.DataFrame,
    *,
    grid_limit_kw: float = 15.0,
    panel_limit_kw: float = 38.0,
) -> Scenario3DOECompliance:
    """Measure violations of the study's two distinct Scenario 3 limits.

    The utility envelope measures net grid import only in its weekday peak
    windows. The panel-loading proxy measures positive household consumption
    before any V2H offset throughout the simulation. Interval violations use
    complete, clock-aligned 15-minute mean powers; minute violations are also
    reported because those intervals can hide shorter spikes.
    """
    if not isinstance(data.index, pd.DatetimeIndex) or data.empty:
        raise ValueError("Scenario 3 compliance needs nonempty minute-indexed data")
    if not data.index.is_monotonic_increasing or data.index.has_duplicates:
        raise ValueError("Scenario 3 minute timestamps must be ordered and unique")
    if data.index[0] != data.index[0].floor("15min") or len(data) % 15:
        raise ValueError("Scenario 3 compliance needs complete aligned 15-minute blocks")
    expected = pd.date_range(data.index[0], periods=len(data), freq="min")
    if not data.index.equals(expected):
        raise ValueError("Scenario 3 compliance needs consecutive one-minute readings")
    for name, value in (("grid_limit_kw", grid_limit_kw), ("panel_limit_kw", panel_limit_kw)):
        if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    required = ("total_electric_power_kw", "ev_power_kw")
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(f"Scenario 3 compliance needs columns: {', '.join(missing)}")

    net = data["total_electric_power_kw"].astype(float)
    ev = data["ev_power_kw"].astype(float)
    if not np.isfinite(net.to_numpy()).all() or not np.isfinite(ev.to_numpy()).all():
        raise ValueError("Scenario 3 power readings must be finite")
    panel = net + (-ev).clip(lower=0)
    minute_active = pd.Series(_peak_mask(data.index), index=data.index)
    minute_grid_excess = (net - grid_limit_kw).clip(lower=0).where(minute_active, 0)
    minute_panel_excess = (panel - panel_limit_kw).clip(lower=0)

    intervals = pd.DataFrame({
        "net_grid_kw": net.resample("15min").mean(),
        "panel_gross_kw": panel.resample("15min").mean(),
    })
    intervals["grid_envelope_active"] = _peak_mask(intervals.index)
    intervals["grid_excess_kw"] = (
        (intervals.net_grid_kw - grid_limit_kw).clip(lower=0)
        .where(intervals.grid_envelope_active, 0)
    )
    intervals["panel_excess_kw"] = (intervals.panel_gross_kw - panel_limit_kw).clip(lower=0)
    active_grid = intervals.loc[intervals.grid_envelope_active, "net_grid_kw"]
    active_grid_minutes = net.loc[minute_active]

    summary: dict[str, float | int | None] = {
        "grid_limit_kw": float(grid_limit_kw),
        "panel_limit_kw": float(panel_limit_kw),
        "interval_minutes": 15,
        "grid_active_intervals": int(intervals.grid_envelope_active.sum()),
        "grid_exceeding_intervals": int(intervals.grid_excess_kw.gt(0).sum()),
        "grid_excess_energy_kwh": float(intervals.grid_excess_kw.sum() * 0.25),
        "grid_active_15min_peak_kw": float(active_grid.max()) if len(active_grid) else None,
        "panel_intervals": len(intervals),
        "panel_exceeding_intervals": int(intervals.panel_excess_kw.gt(0).sum()),
        "panel_excess_energy_kwh": float(intervals.panel_excess_kw.sum() * 0.25),
        "panel_15min_peak_kw": float(intervals.panel_gross_kw.max()),
        "grid_active_one_minute_peak_kw": (
            float(active_grid_minutes.max()) if len(active_grid_minutes) else None
        ),
        "grid_exceeding_one_minute_readings": int(minute_grid_excess.gt(0).sum()),
        "panel_one_minute_peak_kw": float(panel.max()),
        "panel_exceeding_one_minute_readings": int(minute_panel_excess.gt(0).sum()),
    }
    return Scenario3DOECompliance(intervals=intervals, summary=summary)
