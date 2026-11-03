"""Prepare the Outside-Hub target for the existing hybrid load forecaster.

This is an offline, quality-flagged source series. Its test-period readings
are never optimizer inputs: ``RestOfHomeHybridForecast`` fits before the
source replay starts and permits observations only after they are ready at a
forecast issue. The separate OCHRE plant replay performs its approved final
per-meter fill; that future-dependent fill is not copied into this file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from examples.single_home_hems.build_dc_house_rest_of_home import (
    DATA_DIR,
    METER_MISMATCH_INTERVALS,
)

from .home_hub_replay import (
    METER_NAMES,
    StageOneHomeHubMeters,
    load_stage_one_home_hub_meters,
    split_aligned_home_hub_loads,
)


def outside_hub_forecast_minutes(stage_one: StageOneHomeHubMeters) -> pd.DataFrame:
    """Return interval-end source watts with the established meter quality rule.

    A minute is observed only if all six aligned source meters were observed;
    it is interpolated if any source meter used the existing bounded/stable
    fill. Invalid split balances, missing meters, and the established total-
    home meter mismatch windows remain excluded. This mirrors the original
    rest-of-home preparation while replacing its target by Total minus Hub.
    """
    aligned = stage_one.aligned_w
    meter_quality = stage_one.quality
    if not aligned.index.equals(meter_quality.index):
        raise ValueError("Outside-Hub meter values and quality must share one timestamp grid")
    if set(aligned.columns) != set(METER_NAMES) or set(meter_quality.columns) != set(METER_NAMES):
        raise ValueError("Outside-Hub preparation requires all six source meters")
    split = split_aligned_home_hub_loads(aligned)
    observed = meter_quality.eq("observed").all(axis=1)
    valid = split["valid"].copy()
    mismatch = pd.Series(False, index=aligned.index)
    for start, end in METER_MISMATCH_INTERVALS:
        mismatch |= aligned.index.to_series().between(pd.Timestamp(start), pd.Timestamp(end))
    valid &= ~mismatch

    quality = pd.Series("observed", index=aligned.index, dtype="object")
    quality.loc[valid & ~observed] = "interpolated"
    quality.loc[~valid & split["missing_meter"]] = "excluded_missing_meter"
    quality.loc[~valid & mismatch] = "excluded_meter_mismatch"
    quality.loc[~valid & ~split["missing_meter"]
                & (split["negative_hub_other"] | split["negative_outside_hub"])] = (
        "excluded_negative_balance"
    )
    if (quality.str.startswith("excluded") != ~valid).any():
        raise AssertionError("Outside-Hub quality does not match source validity")
    value = split["outside_hub_raw_w"].where(valid)
    if not np.isfinite(value.dropna().to_numpy(dtype=float)).all() or value.dropna().lt(0).any():
        raise ValueError("Valid Outside-Hub source power must be finite and nonnegative")
    return pd.DataFrame({
        "Time": aligned.index,
        "Value": value.to_numpy(dtype=float),
        "Quality": quality.to_numpy(dtype=object),
    })


def build_outside_hub_forecast_csv(
    output_path: str | Path, *, data_dir: str | Path = DATA_DIR,
) -> Path:
    """Write the Outside-Hub target accepted by ``RestOfHomeHybridForecast``.

    The source series covers the full meter history, including pre-test
    training and calibration days. Missing test minutes remain missing here;
    callers may select the forecaster's existing ``causal_short_gap`` option
    for inference, which waits for the later observed gap endpoint.
    """
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    minutes = outside_hub_forecast_minutes(
        load_stage_one_home_hub_meters(data_dir)
    )
    minutes.to_csv(target, index=False, float_format="%.9f",
                   date_format="%Y-%m-%d %H:%M:%S")
    return target
