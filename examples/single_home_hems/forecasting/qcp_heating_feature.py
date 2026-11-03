"""Approved hour-pooled QCP heating estimate for simulated Home Hub forecasts.

This variant keeps the existing QCP empirical CDF, 1 C bins, nearest-five
sparse-bin rule, and constrained three-section curve.  It pools weekdays and
weekends within each local clock hour, giving 24 groups instead of the AC
study's 48.  For an *observed, completed* hour, its heating estimate is the
hourly Home Hub-minus-EV energy minus the fitted base when outdoor temperature
is below the fitted heating change point, and zero otherwise.  As in the AC
equation, the difference is not clipped to zero.

Only aggregate Hub power, the separately metered EV, and observed outdoor
temperature are model inputs.  Hidden OCHRE heat-pump/fan power may be used
later as an evaluation label but is never read here.  A QCP feature issued at
a forecast origin uses the latest completed hour whose reading has arrived.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import pandas as pd

from .qcp_disaggregation import QCPSurface
from .qcp_observed_features import (
    IssuedQCPCurveFeatures,
    issue_qcp_curve_features,
)


LOCAL_TZ = "America/Indiana/Indianapolis"
INTERVAL = pd.Timedelta(minutes=15)
HOUR = pd.Timedelta(hours=1)
DAY = pd.Timedelta(days=1)
MIN_POOL_SAMPLES = 5
FEATURE_COLUMN = "qcp_heating_feature_kw"
READY_COLUMN = "qcp_feature_ready_at"


def _local_timestamp(value: pd.Timestamp | str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("QCP timestamps cannot be missing")
    return stamp.tz_localize(LOCAL_TZ) if stamp.tzinfo is None else stamp.tz_convert(LOCAL_TZ)


def _local_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if index.tz is None:
        return index.tz_localize(LOCAL_TZ)
    return index.tz_convert(LOCAL_TZ)


def hourly_from_baseline(baseline: pd.DataFrame) -> pd.DataFrame:
    """Aggregate complete 15-minute Hub-minus-EV and weather observations.

    The baseline index marks interval starts.  One hourly energy in kWh is the
    mean of four 15-minute kW blocks; the final block becomes available at the
    hour end.  A caller may pass only the public aggregate/EV/weather columns.
    """
    if not isinstance(baseline.index, pd.DatetimeIndex):
        raise ValueError("Baseline needs a DatetimeIndex of 15-minute starts")
    required = {"home_hub_power_kw", "ev_power_kw", "ambient_temperature_c"}
    missing = required - set(baseline)
    if missing:
        raise ValueError(f"QCP source baseline lacks {sorted(missing)}")
    local = _local_index(baseline.index)
    if local.has_duplicates or not local.is_monotonic_increasing:
        raise ValueError("QCP baseline timestamps must be unique and increasing")
    if len(local) > 1 and not (local[1:] - local[:-1] == INTERVAL).all():
        raise ValueError("QCP baseline needs an unbroken 15-minute grid")
    hub = pd.to_numeric(baseline["home_hub_power_kw"], errors="raise").to_numpy(float)
    ev = pd.to_numeric(baseline["ev_power_kw"], errors="raise").to_numpy(float)
    weather = pd.to_numeric(baseline["ambient_temperature_c"], errors="raise").to_numpy(float)
    non_ev = hub - ev
    if not np.isfinite([hub, ev, weather]).all() or (non_ev < 0).any():
        raise ValueError("QCP source power/weather must be finite with nonnegative non-EV Hub load")
    blocks = pd.DataFrame({
        "total_kw": non_ev,
        "temperature_c": weather,
    }, index=local)
    groups = blocks.groupby(blocks.index.floor("h"), sort=True)
    hourly = groups.agg(
        total_energy_kwh=("total_kw", "mean"),
        temperature_c=("temperature_c", "mean"),
        block_count=("total_kw", "size"),
    )
    if (hourly["block_count"] != 4).any():
        raise ValueError("Every QCP source hour needs four complete quarter-hour blocks")
    expected_starts = hourly.index.repeat(4) + np.tile(
        np.arange(4) * INTERVAL, len(hourly)
    )
    if not local.equals(pd.DatetimeIndex(expected_starts)):
        raise ValueError("QCP source blocks must start at 00, 15, 30, and 45 minutes")
    hourly.index.name = "hour_start_local"
    hourly["hour_end_utc"] = (hourly.index + HOUR).tz_convert("UTC")
    hourly["ready_at_utc"] = hourly["hour_end_utc"]
    return hourly.drop(columns="block_count")


@dataclass(frozen=True)
class HourPooledQCP:
    """The approved 24 local-hour QCP surfaces and their fit provenance."""

    surfaces: dict[int, QCPSurface]
    fit_start_local: pd.Timestamp
    fit_cutoff_utc: pd.Timestamp
    summary: dict[str, int]

    def surface_at(self, timestamp: pd.Timestamp | str) -> QCPSurface:
        local = _local_timestamp(timestamp)
        if local.minute or local.second or local.microsecond or local.nanosecond:
            raise ValueError("QCP groups require an hourly local timestamp")
        return self.surfaces[local.hour]


class QCPFitUnavailable(ValueError):
    """Approved QCP group/data criteria are unmet in a short training window."""

    def __init__(self, message: str, coverage: dict[str, int]) -> None:
        super().__init__(message)
        self.coverage = coverage


def fit_hour_pooled_qcp(
    hourly: pd.DataFrame,
    start: pd.Timestamp | str,
    cutoff: pd.Timestamp | str,
    *,
    min_pool_samples: int = MIN_POOL_SAMPLES,
) -> HourPooledQCP:
    """Fit only available complete hours in ``[start, cutoff)``.

    Exact 1 C cells with at least five samples keep all their observations.
    Every other represented integer cell uses the nearest five source hours
    within the same hour-of-day group; ties preserve source timestamp order.
    All 24 hour groups need at least five observations and three represented
    integer temperature bins, counting intervening sparse bins just as the
    original pooling helper does.  No weekday/weekend partition is used.
    """
    if min_pool_samples < 2:
        raise ValueError("min_pool_samples must be at least two")
    if not isinstance(hourly.index, pd.DatetimeIndex) or hourly.index.tz is None:
        raise ValueError("QCP hourly data needs a timezone-aware local index")
    start_local = _local_timestamp(start)
    cutoff_local = _local_timestamp(cutoff)
    cutoff_utc = cutoff_local.tz_convert("UTC")
    if cutoff_local <= start_local:
        raise ValueError("QCP fit cutoff must follow its start")
    required = {"total_energy_kwh", "temperature_c", "hour_end_utc", "ready_at_utc"}
    missing = required - set(hourly)
    if missing:
        raise ValueError(f"QCP fit data lacks {sorted(missing)}")
    end_utc = pd.to_datetime(hourly["hour_end_utc"], utc=True)
    ready_utc = pd.to_datetime(hourly["ready_at_utc"], utc=True)
    selected = hourly.loc[
        (hourly.index >= start_local) & (hourly.index < cutoff_local)
        & (end_utc <= cutoff_utc) & (ready_utc <= cutoff_utc)
    ]
    if selected.empty:
        raise ValueError("No complete available QCP training hours")
    if selected.index.has_duplicates or not selected.index.is_monotonic_increasing:
        raise ValueError("QCP fit timestamps must be unique and ordered")
    total = selected["total_energy_kwh"].to_numpy(float)
    temperature = selected["temperature_c"].to_numpy(float)
    if not np.isfinite(total).all() or not np.isfinite(temperature).all() or (total < 0).any():
        raise ValueError("QCP fit needs finite nonnegative energy and finite weather")

    coverage: dict[str, int] = {
        "training_hours": len(selected),
        "hour_groups": 24,
        "groups_with_min_samples": 0,
        "groups_with_three_represented_bins": 0,
        "eligible_hour_groups": 0,
        "min_pool_samples": min_pool_samples,
    }
    for hour in range(24):
        rows = selected.loc[selected.index.hour == hour]
        enough_samples = len(rows) >= min_pool_samples
        source_bins = np.floor(rows["temperature_c"].to_numpy(float))
        represented_bins = int(source_bins.max() - source_bins.min() + 1) if len(source_bins) else 0
        enough_bins = represented_bins >= 3
        coverage["groups_with_min_samples"] += int(enough_samples)
        coverage["groups_with_three_represented_bins"] += int(enough_bins)
        coverage["eligible_hour_groups"] += int(enough_samples and enough_bins)
    if coverage["eligible_hour_groups"] < 24:
        raise QCPFitUnavailable(
            f"QCP fit has {coverage['eligible_hour_groups']}/24 eligible hour groups "
            f"({coverage['groups_with_min_samples']}/24 with {min_pool_samples} samples, "
            f"{coverage['groups_with_three_represented_bins']}/24 with three represented 1 C bins)",
            coverage,
        )

    surfaces: dict[int, QCPSurface] = {}
    exact_cells = 0
    pooled_cells = 0
    for hour in range(24):
        rows = selected.loc[selected.index.hour == hour]
        degrees = rows["temperature_c"].to_numpy(float)
        energy = rows["total_energy_kwh"].to_numpy(float)
        source_bins = np.floor(degrees).astype(int)
        first, last = int(source_bins.min()), int(source_bins.max())
        bins: dict[int, np.ndarray] = {}
        for bin_c in range(first, last + 1):
            exact = energy[source_bins == bin_c]
            if len(exact) >= min_pool_samples:
                values = exact
                exact_cells += 1
            else:
                nearest = np.argsort(
                    np.abs(degrees - (bin_c + 0.5)), kind="stable"
                )[:min_pool_samples]
                values = energy[nearest]
                pooled_cells += 1
            bins[bin_c] = np.sort(values)
        surfaces[hour] = QCPSurface.from_temperature_bins(bins)
    return HourPooledQCP(
        surfaces=surfaces,
        fit_start_local=start_local,
        fit_cutoff_utc=cutoff_utc,
        summary={
            **coverage,
            "exact_temperature_cells": exact_cells,
            "pooled_temperature_cells": pooled_cells,
        },
    )


def try_fit_hour_pooled_qcp(
    hourly: pd.DataFrame,
    start: pd.Timestamp | str,
    cutoff: pd.Timestamp | str,
    *,
    min_pool_samples: int = MIN_POOL_SAMPLES,
) -> tuple[HourPooledQCP | None, dict[str, object]]:
    """Return explicit coverage when the approved sparse-window fit cannot run."""
    try:
        model = fit_hour_pooled_qcp(
            hourly, start, cutoff, min_pool_samples=min_pool_samples
        )
    except QCPFitUnavailable as exc:
        return None, {"fit_available": False, "reason": str(exc), **exc.coverage}
    return model, {"fit_available": True, **model.summary}


@dataclass(frozen=True)
class IssuedQCPHeatingFeature:
    """Observed-hour heating estimate with its measurement provenance."""

    heating_feature_kw: float
    ready_at_local: pd.Timestamp
    source: IssuedQCPCurveFeatures


def issue_qcp_heating_feature(
    model: HourPooledQCP,
    hourly: pd.DataFrame,
    origin: pd.Timestamp | str,
) -> IssuedQCPHeatingFeature:
    """Issue the approved symmetric heating estimate without clipping."""
    source = issue_qcp_curve_features(
        model, hourly, _local_timestamp(origin), model_fit_end=model.fit_cutoff_utc,
    )
    estimate = (
        source.source_total_energy_kwh - source.base_kw
        if source.source_temperature_c < source.heating_change_point_c else 0.0
    )
    if not math.isfinite(estimate):
        raise ValueError("QCP heating estimate is nonfinite")
    return IssuedQCPHeatingFeature(
        heating_feature_kw=estimate,
        ready_at_local=max(
            source.source_ready_at_utc, model.fit_cutoff_utc
        ).tz_convert(LOCAL_TZ),
        source=source,
    )


def features_at_origins(
    model: HourPooledQCP | None,
    hourly: pd.DataFrame,
    origins: Iterable[pd.Timestamp | str],
) -> pd.DataFrame:
    """Return the two optional columns accepted by the Hub ML benchmark.

    The ready timestamps use the local wall clock and the returned index keeps
    the caller's origin representation.  Origins before the first complete
    observation receive a missing feature; other fit/data failures propagate.
    """
    index = pd.DatetimeIndex(origins, name="timestamp")
    if model is None:
        return pd.DataFrame({
            FEATURE_COLUMN: np.full(len(index), np.nan),
            READY_COLUMN: pd.Series(pd.NaT, index=index, dtype="datetime64[ns]"),
        }, index=index)
    values: list[float] = []
    ready: list[pd.Timestamp | pd.NaT] = []
    for origin in index:
        if _local_timestamp(origin).tz_convert("UTC") < model.fit_cutoff_utc:
            raise ValueError("QCP fit includes labels after the forecast origin")
        if hourly.empty or (hourly.index[0] + HOUR) > _local_timestamp(origin):
            values.append(np.nan)
            ready.append(pd.NaT)
            continue
        issued = issue_qcp_heating_feature(model, hourly, origin)
        values.append(issued.heating_feature_kw)
        ready.append(issued.ready_at_local.tz_localize(None))
    return pd.DataFrame({FEATURE_COLUMN: values, READY_COLUMN: ready}, index=index)


def walk_forward_training_features(
    hourly: pd.DataFrame,
    origins: Iterable[pd.Timestamp | str],
    training_start: pd.Timestamp | str,
    *,
    min_pool_samples: int = MIN_POOL_SAMPLES,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Create training features with a QCP fit from strictly earlier days.

    Each local day uses a QCP fit on available hours from ``training_start``
    through midnight preceding that day.  Early days where the approved
    24-group fit is infeasible receive a missing feature.  Those rows remain
    usable by the forecast benchmark through its availability indicator.
    """
    index = pd.DatetimeIndex(origins, name="timestamp")
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("Training QCP origins must be unique and increasing")
    result = pd.DataFrame({FEATURE_COLUMN: np.nan, READY_COLUMN: pd.NaT}, index=index)
    start = _local_timestamp(training_start)
    failures: dict[str, str] = {}
    fitted_days = 0
    for day in _local_index(index).normalize().unique():
        mask = _local_index(index).normalize() == day
        if day <= start:
            continue
        try:
            model = fit_hour_pooled_qcp(
                hourly, start, day, min_pool_samples=min_pool_samples
            )
        except QCPFitUnavailable as exc:
            failures[str(day.date())] = str(exc)
            continue
        daily = features_at_origins(model, hourly, index[mask])
        result.loc[index[mask], [FEATURE_COLUMN, READY_COLUMN]] = daily.to_numpy()
        fitted_days += 1
    return result, {
        "fitted_days": fitted_days,
        "unavailable_days": failures,
        "available_origins": int(result[FEATURE_COLUMN].notna().sum()),
        "total_origins": len(index),
    }
