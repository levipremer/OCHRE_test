"""Causal multiresolution aggregate-load history for the EV-only RF study.

The source Outside-Hub meter timestamps mark *ends* of one-minute intervals.
Its observation at simulation time ``t`` covers ``(t - 1 minute, t]``. The
OCHRE electric-resistance channel is recorded at the *start* of that same
interval. It is combined with the fixed Outside-Hub meter only to reconstruct
the aggregate meter that the EV-only controller may observe; it is not a
separate model feature.

For an issue at ``t``, the 44 values partition the preceding 24 hours:
15 individual minutes ending ``t`` through ``t - 14 minutes``; seven complete
15-minute means from 15 to 120 minutes old; and 22 complete hourly means
from two to 24 hours old. Every value has its own availability flag. A block
is available only when all of its original source minutes were observed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .evaluate_outside_hub import (
        BASELINE_DIR, DEFAULT_OUTPUT as FIXED_STUDY_DIR, SIMULATION_START,
    )
    from .outside_hub_reference import SOURCE_START
except ImportError:  # direct script invocation from the repository root
    from evaluate_outside_hub import (
        BASELINE_DIR, DEFAULT_OUTPUT as FIXED_STUDY_DIR, SIMULATION_START,
    )
    from outside_hub_reference import SOURCE_START


MINUTE = pd.Timedelta(minutes=1)
FIFTEEN_MINUTES = pd.Timedelta(minutes=15)
ER_COLUMN = "HVAC Heating ER Power (kW)"
DEFAULT_SOURCE = FIXED_STUDY_DIR / "outside_hub_forecast_source.csv"
DEFAULT_RAW = BASELINE_DIR / "data/raw/home_01.csv"
BALANCE_TOLERANCE_KW = 1e-6


def _target_grid(truth: pd.Series, observed: pd.Series) -> None:
    """Require matching, left-labelled complete 15-minute evaluation bins."""
    index = truth.index
    if (not isinstance(index, pd.DatetimeIndex) or index.tz is not None
            or index.hasnans or index.has_duplicates
            or not index.equals(observed.index) or len(index) == 0
            or not index.equals(pd.date_range(
                SIMULATION_START, periods=len(index), freq="15min", name=index.name,
            ))):
        raise ValueError("Truth and observed targets need the same regular simulation 15-minute grid")
    if not np.isfinite(truth.to_numpy(float)).all():
        raise ValueError("Replay truth must be finite on every 15-minute bin")
    known = observed.dropna().to_numpy(float)
    if not np.isfinite(known).all():
        raise ValueError("Observed target bins must be finite where available")


def load_minute_observed(
    truth: pd.Series,
    observed: pd.Series,
    *,
    source_path: str | Path = DEFAULT_SOURCE,
    raw_path: str | Path = DEFAULT_RAW,
) -> tuple[pd.Series, dict[str, Any]]:
    """Reconstruct observed aggregate one-minute power, timestamped at interval end.

    ``truth`` is the full 15-minute Outside-Hub-plus-resistance replay and
    ``observed`` is that target masked to originally observed 15-minute bins.
    They are used to audit the reconstruction, never to fill missing minutes.
    Source ``Value`` is in W and the result is in kW. The returned series has
    NaN at every minute whose source ``Quality`` is not ``observed``.
    """
    _target_grid(truth, observed)
    source_path = Path(source_path).expanduser().resolve()
    raw_path = Path(raw_path).expanduser().resolve()
    source = pd.read_csv(
        source_path, usecols=["Time", "Value", "Quality"], parse_dates=["Time"],
    )
    times = pd.DatetimeIndex(source["Time"])
    if (times.tz is not None or times.hasnans or times.has_duplicates
            or not times.is_monotonic_increasing):
        raise ValueError("Outside-Hub source minute timestamps must be unique, naive, and increasing")
    shift = SIMULATION_START - SOURCE_START
    source.index = pd.DatetimeIndex(times + shift, name="timestamp")
    expected_ends = pd.date_range(
        truth.index[0] + MINUTE, periods=len(truth) * 15,
        freq="min", name="timestamp",
    )
    source = source.reindex(expected_ends)
    if source["Time"].isna().any():
        raise ValueError("Outside-Hub source does not cover every evaluation minute")
    values_w = pd.to_numeric(source["Value"], errors="raise")
    if (not np.isfinite(values_w.dropna().to_numpy(float)).all()
            or values_w.dropna().lt(0).any()):
        raise ValueError("Outside-Hub source values must be finite and nonnegative")
    quality = source["Quality"].astype(str)
    if (quality.eq("observed") & values_w.isna()).any():
        raise ValueError("An observed Outside-Hub minute lacks its power reading")

    raw = pd.read_csv(
        raw_path, usecols=["Time", ER_COLUMN], parse_dates=["Time"],
    )
    raw_times = pd.DatetimeIndex(raw["Time"], name="timestamp")
    if (raw_times.tz is not None or raw_times.hasnans or raw_times.has_duplicates
            or not raw_times.is_monotonic_increasing):
        raise ValueError("OCHRE electric-resistance timestamps must be unique, naive, and increasing")
    er = pd.Series(
        pd.to_numeric(raw[ER_COLUMN], errors="raise").to_numpy(float),
        index=raw_times,
    ).reindex(expected_ends - MINUTE)
    er_values = er.to_numpy(float)
    if (not np.isfinite(er_values).all()
            or np.min(er_values) < -BALANCE_TOLERANCE_KW):
        raise ValueError("OCHRE electric resistance must be complete and nonnegative")
    aggregate = pd.Series(
        values_w.to_numpy(float) / 1000.0 + np.maximum(er_values, 0.0),
        index=expected_ends,
        name="outside_hub_with_hvac_er_observed_kw",
    ).where(quality.eq("observed").to_numpy())

    # Right-labelled one-minute intervals form left-labelled 15-minute bins:
    # end stamps 00:01–00:15 belong to the target bin labelled 00:00.
    grouped = aggregate.resample("15min", closed="right", label="right")
    reconstructed = grouped.mean().where(grouped.count().eq(15))
    reconstructed.index = reconstructed.index - FIFTEEN_MINUTES
    reconstructed = reconstructed.reindex(truth.index)
    known = observed.notna()
    if not reconstructed.notna().equals(known):
        raise ValueError("Minute observation mask disagrees with original 15-minute target mask")
    mismatch_observed = float(np.max(np.abs(
        reconstructed.loc[known].to_numpy(float)
        - observed.loc[known].to_numpy(float)
    ))) if known.any() else 0.0
    mismatch_truth = float(np.max(np.abs(
        reconstructed.loc[known].to_numpy(float)
        - truth.loc[known].to_numpy(float)
    ))) if known.any() else 0.0
    if max(mismatch_observed, mismatch_truth) > BALANCE_TOLERANCE_KW:
        raise ValueError("Observed aggregate minute power disagrees with the 15-minute target")
    audit = {
        "source_path": str(source_path),
        "raw_ochre_path": str(raw_path),
        "minute_timestamp": "interval end",
        "ochre_er_alignment": "one minute before source interval end",
        "source_quality": "observed only",
        "observed_minutes": int(aggregate.notna().sum()),
        "unavailable_minutes": int(aggregate.isna().sum()),
        "observed_15min_bins": int(known.sum()),
        "max_observed_bin_error_kw": mismatch_observed,
        "max_truth_bin_error_kw": mismatch_truth,
    }
    return aggregate, audit


def load_feature_matrix(
    minute_observed: pd.Series,
    origins: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Return 44 aggregate values and 44 flags, using only readings ready at issue.

    The most recent reading has an interval-end timestamp equal to the issue.
    Block means use nonoverlapping, complete original observed minutes. Missing
    readings/blocks receive value zero and availability zero; a truly zero
    observed block receives value zero and availability one.
    """
    index = minute_observed.index
    if (not isinstance(index, pd.DatetimeIndex) or index.tz is not None
            or index.hasnans or index.has_duplicates or len(index) == 0
            or not index.equals(pd.date_range(index[0], periods=len(index),
                                               freq="min", name=index.name))):
        raise ValueError("Observed minute power needs a complete regular naive one-minute grid")
    if (not isinstance(origins, pd.DatetimeIndex) or origins.tz is not None
            or origins.hasnans):
        raise ValueError("Forecast origins need naive minute-aligned timestamps")
    values = minute_observed.to_numpy(float)
    if np.isinf(values).any():
        raise ValueError("Minute observations must be finite where available")
    minute_ns = MINUTE.value
    differences = origins.asi8 - index[0].value
    if np.any(differences % minute_ns):
        raise ValueError("Forecast origins must align with the minute meter grid")
    positions = differences // minute_ns
    observed = np.isfinite(values)
    sums = np.concatenate(([0.0], np.cumsum(np.where(observed, values, 0.0))))
    counts = np.concatenate(([0], np.cumsum(observed.astype(np.int64))))
    columns: dict[str, np.ndarray] = {}
    n = len(values)

    def add(name: str, newest_age_minutes: int, width_minutes: int) -> None:
        hi = positions - newest_age_minutes
        lo = hi - width_minutes + 1
        in_bounds = (lo >= 0) & (hi < n)
        lo_safe = np.clip(lo, 0, n)
        hi_plus_one_safe = np.clip(hi + 1, 0, n)
        available = in_bounds & (
            counts[hi_plus_one_safe] - counts[lo_safe] == width_minutes
        )
        block_sum = sums[hi_plus_one_safe] - sums[lo_safe]
        columns[f"{name}_kw"] = np.where(
            available, block_sum / width_minutes, 0.0,
        )
        columns[f"{name}_available"] = available.astype(np.int8)

    for age in range(15):
        add(f"load_minute_lag_{age}", age, 1)
    for block in range(7):
        add(f"load_prev_15m_block_{block + 1}", 15 + 15 * block, 15)
    for block in range(22):
        add(f"load_prev_1h_block_{block + 1}", 120 + 60 * block, 60)
    result = pd.DataFrame(columns, index=origins)
    if result.shape != (len(origins), 88):
        raise AssertionError("Expected 44 load readings/means and 44 availability flags")
    return result
