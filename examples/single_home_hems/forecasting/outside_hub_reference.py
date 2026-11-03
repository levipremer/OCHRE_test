"""Issue the unchanged Outside-Hub RF-to-p90 controller reference.

The source is the existing quality-flagged Total-minus-Hub meter series. Its
originally observed bins fit the existing 20-day conservative RF, while the
90th-percentile clock profile uses the prior 20 completed source days. The
controller's causal short-gap inference policy is extended to the six-week
source window. Missing latest readings are reported as unavailable issues.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from examples.single_home_hems.single_home_hems.rest_of_home_forecast import (
    RestOfHomeHybridForecast,
)


SIMULATION_START = pd.Timestamp("2018-01-07 00:00:00")
SOURCE_START = pd.Timestamp("2026-08-30 00:00:00")
INTERVAL = pd.Timedelta(minutes=15)
HORIZON_STEPS = 32
SOURCE_DAYS = 42
METHOD = "existing_rf_p90_20d"
_MISSING_LATEST_PREFIX = "Conservative RF needs the latest approved 15-minute bin available by "


def _checked_origins(origins: Iterable[pd.Timestamp]) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(origins, name="origin")
    if (index.empty or index.hasnans or index.tz is not None
            or index.has_duplicates or not index.is_monotonic_increasing
            or not index.equals(index.floor("15min"))):
        raise ValueError("Reference origins must be unique, increasing naive 15-minute times")
    return index


def issue_reference(
    *,
    source_path: str | Path,
    origins: Iterable[pd.Timestamp],
    actual_at_target: pd.Series,
    simulation_start: pd.Timestamp = SIMULATION_START,
    source_start: pd.Timestamp = SOURCE_START,
    split: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Issue the existing 8-hour forecast from a prebuilt deployment source.

    Returns long predictions, one diagnostic per unavailable issue, and the
    fitted provider's metadata. Only the provider's known missing-latest-bin
    error is recorded as an unavailable issue; every other exception rises.
    The caller supplies common scored origins, the completed baseline target,
    and a source CSV built by ``build_outside_hub_forecast_csv``. This function
    fits the provider exactly once at the baseline start.
    """
    selected = _checked_origins(origins)
    if (not isinstance(actual_at_target, pd.Series)
            or not isinstance(actual_at_target.index, pd.DatetimeIndex)
            or actual_at_target.index.hasnans or actual_at_target.index.tz is not None
            or actual_at_target.index.has_duplicates
            or not actual_at_target.index.is_monotonic_increasing):
        raise ValueError("Reference actuals require an increasing, unique naive timestamp series")
    provider = RestOfHomeHybridForecast.from_csv(
        source_path,
        simulation_start=simulation_start,
        source_start=source_start,
        history_days=20,
        inference_gap_policy="causal_short_gap",
        inference_window=(pd.Timestamp(source_start),
                          pd.Timestamp(source_start) + pd.Timedelta(days=SOURCE_DAYS)),
    )
    predictions: list[pd.DataFrame] = []
    unavailable: list[dict[str, object]] = []
    last_minute_hold_issues = 0
    plant_fill_issues = 0
    for origin in selected:
        try:
            issued = provider.forecast_kw(origin, steps=HORIZON_STEPS)
        except ValueError as exc:
            if not str(exc).startswith(_MISSING_LATEST_PREFIX):
                raise
            source_origin = provider.source_timestamp(origin)
            latest_bin = source_origin - INTERVAL
            unavailable.append({
                "split": split,
                "origin": origin,
                "source_origin": source_origin,
                "latest_source_bin": latest_bin,
                "latest_approved_w": provider.approved.loc[latest_bin],
                "latest_observed_w": provider.observed.loc[latest_bin],
                "latest_ready_at": provider.approved_ready_at.loc[latest_bin],
                "reason": str(exc),
            })
            continue
        target = pd.date_range(origin, periods=HORIZON_STEPS, freq=INTERVAL)
        if (not issued.index.equals(target) or not np.isfinite(issued.to_numpy(float)).all()
                or issued.lt(0).any()):
            raise AssertionError("Existing Outside-Hub reference returned invalid intervals")
        last_minute_hold_issues += int(issued.attrs["rf_latest_bin_last_minute_hold"])
        plant_fill_issues += int(issued.attrs["rf_plant_fill_input_bins"] > 0)
        actual = actual_at_target.reindex(target).to_numpy(dtype=float)
        if not np.isfinite(actual).all() or (actual < 0).any():
            raise ValueError(f"Reference lacks complete nonnegative truth at {origin}")
        predictions.append(pd.DataFrame({
            "method": METHOD,
            "train_weeks": np.nan,
            "split": split,
            "origin": origin,
            "target": target,
            "lead": np.arange(1, HORIZON_STEPS + 1),
            "forecast_kw": issued.to_numpy(float),
            "actual_kw": actual,
        }))
    columns = [
        "method", "train_weeks", "split", "origin", "target", "lead",
        "forecast_kw", "actual_kw",
    ]
    diagnostic_columns = [
        "split", "origin", "source_origin", "latest_source_bin", "latest_approved_w",
        "latest_observed_w", "latest_ready_at", "reason",
    ]
    long = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(columns=columns)
    missing = pd.DataFrame(unavailable, columns=diagnostic_columns)
    metadata = {
        **provider.metadata,
        "inference_window_start": pd.Timestamp(source_start).isoformat(),
        "inference_window_end_exclusive": (
            pd.Timestamp(source_start) + pd.Timedelta(days=SOURCE_DAYS)
        ).isoformat(),
        "requested_origins": len(selected),
        "issued_origins": len(predictions),
        "unavailable_origins": len(unavailable),
        "last_minute_hold_issues": last_minute_hold_issues,
        "plant_fill_issues": plant_fill_issues,
    }
    return long, missing, metadata
