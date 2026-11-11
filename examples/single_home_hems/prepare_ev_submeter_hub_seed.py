#!/usr/bin/env python3
# ruff: noqa: E402
"""Prepare causal Home Hub and three-circuit history from native OCHRE results.

Only complete one-minute intervals strictly before the first forecast issue
enter the seed. The corrected Hub excludes EV and HVAC electric resistance;
its circuit definition retains both heating and cooling equipment standby.
No old aggregate seed, missing-value fill, or future interval is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.single_home_hems.single_home_hems.home_hub_observation import (
    CIRCUIT_POWER_COMPONENTS,
)


MINUTE = pd.Timedelta(minutes=1)
BLOCK = pd.Timedelta(minutes=15)
OTHER_KEY = "DC House Hub Other Electric Power (kW)"
TARGET_KEY = "hub_non_ev_power_kw"


def _source_index(values: pd.Series, cutoff: pd.Timestamp) -> tuple[pd.DatetimeIndex, list[pd.Timestamp]]:
    try:
        parsed = [pd.Timestamp(value) for value in values]
    except (ValueError, TypeError) as exc:
        raise ValueError("OCHRE Time contains an invalid timestamp") from exc
    if not parsed or any(pd.isna(value) for value in parsed):
        raise ValueError("OCHRE Time must contain valid timestamps")
    aware = cutoff.tz is not None
    if any((value.tz is not None) != aware for value in parsed):
        raise ValueError("OCHRE timestamps and first issue have mismatched timezone awareness")
    if aware:
        index = pd.DatetimeIndex(pd.to_datetime(values, format="mixed", utc=True)).tz_convert(cutoff.tz)
    else:
        index = pd.DatetimeIndex(parsed)
    return index, parsed


def prepare_ev_submeter_hub_seed(
    raw_file: str | Path,
    first_issue: str | pd.Timestamp,
    output_file: str | Path,
    history_days: int = 7,
) -> dict[str, Any]:
    """Write a new seed CSV and provenance JSON, with no filled source minutes."""
    source = Path(raw_file).expanduser().resolve()
    destination = Path(output_file).expanduser().resolve()
    provenance_path = destination.with_suffix(".provenance.json")
    if destination.exists() or provenance_path.exists():
        raise FileExistsError("Seed CSV and provenance must both be new files")
    if isinstance(history_days, bool) or not isinstance(history_days, int) or history_days < 1:
        raise ValueError("history_days must be a positive integer")
    cutoff = pd.Timestamp(first_issue)
    if pd.isna(cutoff) or cutoff != cutoff.floor("15min"):
        raise ValueError("First issue must be a valid aligned 15-minute timestamp")
    start = cutoff - pd.Timedelta(days=history_days)
    required = list(dict.fromkeys(
        key for components in CIRCUIT_POWER_COMPONENTS.values() for key in components
    )) + [OTHER_KEY]
    headers = pd.read_csv(source, nrows=0).columns
    missing = {"Time", *required} - set(headers)
    if missing:
        raise ValueError(f"Native OCHRE raw source lacks {sorted(missing)}")
    raw = pd.read_csv(source, usecols=["Time", *required])
    index, original_times = _source_index(raw["Time"], cutoff)
    mask = np.asarray((index >= start) & (index < cutoff))
    selected_index = index[mask]
    expected = pd.date_range(start, cutoff, freq=MINUTE, inclusive="left")
    if selected_index.has_duplicates or not selected_index.is_monotonic_increasing:
        raise ValueError("Seed source timestamps must be unique and increasing")
    selected_original_times = [value for value, use in zip(original_times, mask) if use]
    if cutoff.tz is not None and any(
        original.utcoffset() != normalized.utcoffset()
        for original, normalized in zip(selected_original_times, selected_index)
    ):
        raise ValueError("Seed source timezone offsets do not match the first-issue timezone")
    wall = selected_index.tz_localize(None) if selected_index.tz is not None else selected_index
    if wall.has_duplicates or (len(wall) > 1 and not np.all(np.diff(wall.asi8) == MINUTE.value)):
        raise ValueError("Seed window has a missing minute or an ambiguous clock transition")
    if not selected_index.equals(expected):
        missing_minutes = expected.difference(selected_index)
        raise ValueError(f"Seed needs every prior minute; missing {len(missing_minutes)} minute(s)")
    values = raw.loc[mask, required].apply(pd.to_numeric, errors="raise")
    values.index = expected
    if not np.isfinite(values.to_numpy(float)).all() or values.lt(0).any().any():
        raise ValueError("Seed circuit and Hub-other power must be finite and nonnegative")
    circuits = pd.DataFrame({
        alias: values.loc[:, list(components)].sum(axis=1)
        for alias, components in CIRCUIT_POWER_COMPONENTS.items()
    }, index=expected)
    target = circuits.sum(axis=1) + values[OTHER_KEY]
    minute = pd.DataFrame({TARGET_KEY: target, **{key: circuits[key] for key in circuits}},
                          index=expected)
    minute.index.name = "timestamp"
    groups = minute.resample("15min", label="left", closed="left")
    counts = groups.count()
    if not counts.eq(15).all().all():
        raise ValueError("Every seed block must contain exactly 15 observed minutes")
    seed = groups.mean()
    expected_blocks = pd.date_range(start, cutoff, freq=BLOCK, inclusive="left")
    if not seed.index.equals(expected_blocks) or len(seed) != history_days * 96:
        raise ValueError("Seed must contain complete, unambiguous 15-minute blocks")
    other_blocks = values[OTHER_KEY].resample("15min", label="left", closed="left").mean()
    minute_error = (minute[TARGET_KEY] - circuits.sum(axis=1) - values[OTHER_KEY]).abs().max()
    block_error = (seed[TARGET_KEY] - seed[list(CIRCUIT_POWER_COMPONENTS)].sum(axis=1)
                   - other_blocks).abs().max()
    if minute_error > 1e-9 or block_error > 1e-9:
        raise AssertionError("Corrected Hub and circuit meter identity failed")
    report: dict[str, Any] = {
        "source_raw_file": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "first_issue": cutoff.isoformat(),
        "history_start_inclusive": start.isoformat(),
        "history_end_exclusive": cutoff.isoformat(),
        "history_days": history_days,
        "timezone": str(cutoff.tz) if cutoff.tz is not None else "naive OCHRE clock",
        "source_interval_convention": "Time is the start of a completed one-minute interval",
        "output_interval_convention": "timestamp is the start of a complete 15-minute mean",
        "circuit_mapping": {alias: list(keys) for alias, keys in CIRCUIT_POWER_COMPONENTS.items()},
        "hub_other_source": OTHER_KEY,
        "hub_target_definition": "water_heater_kw + outdoor_heat_pump_kw + indoor_fan_kw + Hub-other",
        "excludes_hvac_electric_resistance": True,
        "excludes_ev_power": True,
        "cooling_main_standby_retained": True,
        "old_aggregate_seed_used": False,
        "source_minutes": len(minute),
        "output_blocks": len(seed),
        "future_source_rows_excluded": int((index >= cutoff).sum()),
        "fill_count": 0,
        "identity_checks": {
            "minute_max_absolute_error_kw": float(minute_error),
            "block_max_absolute_error_kw": float(block_error),
            "minutes_per_block": 15,
            "latest_source_interval_end": (expected[-1] + MINUTE).isoformat(),
        },
        "output_file": str(destination),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        seed.to_csv(stream)
    report["output_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
    with provenance_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-file", type=Path, required=True)
    parser.add_argument("--first-issue", required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--history-days", type=int, default=7)
    args = parser.parse_args()
    print(json.dumps(prepare_ev_submeter_hub_seed(
        args.raw_file, args.first_issue, args.output_file, args.history_days,
    ), indent=2))


if __name__ == "__main__":
    main()
