"""Strict measured pre-test split loads for an extended native warm-up.

The October evaluation and all forecast training boundaries remain unchanged.
Only the native plant prelude may use the additional preceding meter readings;
unresolved readings and negative balances are rejected, never padded.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .home_hub_replay import (
    DATA_DIR, FINAL_METER_QUALITY, HOME_HUB_TEST_START, METER_NAMES, MINUTE,
    PreparedHomeHubReplay, load_stage_one_home_hub_meters,
    split_aligned_home_hub_loads,
)


def prepare_home_hub_native_history(
    source_start: str | pd.Timestamp, source_end: str | pd.Timestamp, *,
    data_dir: str | Path = DATA_DIR,
) -> PreparedHomeHubReplay:
    """Read complete measured intervals within the day before October 4.

    Bounds use naive local interval starts and exclude ``source_end``. The
    existing meter alignment and stable-gap treatment are retained. No second
    replay interpolation is applied to this extra native history. In
    particular, this helper does not widen the held-out evaluation window or
    any forecast-training window.
    """
    start, end = pd.Timestamp(source_start), pd.Timestamp(source_end)
    first = HOME_HUB_TEST_START - pd.Timedelta(hours=24)
    if (pd.isna(start) or pd.isna(end) or start.tz is not None or end.tz is not None
            or start != start.floor("min") or end != end.floor("min")
            or not first <= start < end <= HOME_HUB_TEST_START):
        raise ValueError("Native history must use whole-minute local bounds within the 24 hours before October 4")
    stage_one = load_stage_one_home_hub_meters(data_dir)
    ends = pd.date_range(start + MINUTE, end, freq=MINUTE)
    meters = stage_one.aligned_w.reindex(ends)
    quality = stage_one.quality.reindex(ends).fillna("unavailable")
    audit = split_aligned_home_hub_loads(meters)
    if not audit["valid"].all():
        bad = audit.loc[~audit["valid"]]
        raise ValueError(
            f"Native history has {len(bad)} unresolved or negative-balance minute(s); "
            f"first total-meter time {bad.index[0]}, quality {bad.iloc[0]['quality']}"
        )
    starts = pd.DatetimeIndex(audit["source_interval_start"], name="source_interval_start")
    power = pd.DataFrame({
        "hub_other_power_kw": audit["hub_other_w"].to_numpy() / 1000,
        "outside_hub_power_kw": audit["outside_hub_w"].to_numpy() / 1000,
    }, index=starts)
    quality.index = starts
    audit.index = starts
    counts = pd.DataFrame({
        name: quality[name].value_counts().reindex(FINAL_METER_QUALITY, fill_value=0)
        for name in METER_NAMES
    }).T
    counts.index.name = "meter"
    return PreparedHomeHubReplay(power, quality, counts, audit)
