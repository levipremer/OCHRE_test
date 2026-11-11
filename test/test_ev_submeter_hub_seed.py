"""Circuit-history seeds use only complete native minutes before first issue."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from examples.single_home_hems.prepare_ev_submeter_hub_seed import prepare_ev_submeter_hub_seed


FIRST_ISSUE = pd.Timestamp("2025-11-30T00:00:00-05:00")


def raw_history() -> pd.DataFrame:
    index = pd.date_range(FIRST_ISSUE - pd.Timedelta(days=7), periods=7 * 1440 + 30, freq="min")
    water = 1.0 + np.arange(len(index)) / 10000.0
    result = pd.DataFrame({
        "Time": index,
        "Water Heating Electric Power (kW)": water,
        "HVAC Heating Main Power (kW)": 2.0,
        "HVAC Cooling Main Power (kW)": 0.05,
        "HVAC Heating Fan Power (kW)": 0.2,
        "HVAC Cooling Fan Power (kW)": 0.01,
        "DC House Hub Other Electric Power (kW)": 0.3,
        "HVAC Heating ER Power (kW)": 7.0,
        # A legacy aggregate containing ER must not become the new seed target.
        "hub_non_ev_power_kw": water + 2.0 + 0.05 + 0.2 + 0.01 + 0.3 + 7.0,
    })
    return result


def write_raw(tmp_path: Path, frame: pd.DataFrame, name: str = "raw.csv") -> Path:
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path


def test_seed_has_exact_completed_block_means_and_ignores_old_er_aggregate(tmp_path):
    raw = raw_history()
    source = write_raw(tmp_path, raw)
    destination = tmp_path / "seed.csv"
    report = prepare_ev_submeter_hub_seed(source, FIRST_ISSUE, destination)
    seed = pd.read_csv(destination, parse_dates=["timestamp"]).set_index("timestamp")
    assert list(seed.columns) == ["hub_non_ev_power_kw", "water_heater_kw",
                                 "outdoor_heat_pump_kw", "indoor_fan_kw"]
    assert len(seed) == 7 * 96
    assert seed.index[-1] + pd.Timedelta(minutes=15) == FIRST_ISSUE
    assert seed.water_heater_kw.iloc[0] == pytest.approx(raw.iloc[:15]["Water Heating Electric Power (kW)"].mean())
    np.testing.assert_allclose(seed.outdoor_heat_pump_kw, 2.05)
    np.testing.assert_allclose(seed.indoor_fan_kw, 0.21)
    np.testing.assert_allclose(seed.hub_non_ev_power_kw,
                               seed.water_heater_kw + 2.05 + 0.21 + 0.3)
    old = raw.iloc[:7 * 1440].set_index("Time").hub_non_ev_power_kw.resample("15min").mean()
    np.testing.assert_allclose(old.to_numpy() - seed.hub_non_ev_power_kw.to_numpy(), 7.0)
    assert report["source_minutes"] == 10080
    assert report["future_source_rows_excluded"] == 30
    assert report["fill_count"] == 0
    assert report["identity_checks"]["minutes_per_block"] == 15
    assert report["identity_checks"]["latest_source_interval_end"] == FIRST_ISSUE.isoformat()
    assert report["excludes_hvac_electric_resistance"]
    assert report["cooling_main_standby_retained"]
    assert json.loads(destination.with_suffix(".provenance.json").read_text()) == report


def test_future_and_resistance_changes_do_not_change_seed(tmp_path):
    raw = raw_history()
    source = write_raw(tmp_path, raw)
    original = tmp_path / "original_seed.csv"
    prepare_ev_submeter_hub_seed(source, FIRST_ISSUE, original)
    changed = raw.copy()
    changed.loc[changed.Time >= FIRST_ISSUE, "Water Heating Electric Power (kW)"] = np.nan
    changed.loc[changed.Time >= FIRST_ISSUE, "HVAC Heating Main Power (kW)"] = 999.0
    changed["HVAC Heating ER Power (kW)"] = 100.0
    changed["hub_non_ev_power_kw"] = -999.0
    changed_source = write_raw(tmp_path, changed, "changed.csv")
    changed_seed = tmp_path / "changed_seed.csv"
    prepare_ev_submeter_hub_seed(changed_source, FIRST_ISSUE, changed_seed)
    assert changed_seed.read_bytes() == original.read_bytes()


@pytest.mark.parametrize("failure", ["gap", "negative", "nonfinite", "missing_circuit", "duplicate"])
def test_invalid_history_is_rejected_without_fill(tmp_path, failure):
    raw = raw_history()
    if failure == "gap":
        raw = raw.drop(index=23)
    elif failure == "negative":
        raw.loc[23, "HVAC Cooling Main Power (kW)"] = -0.05
    elif failure == "nonfinite":
        raw.loc[23, "HVAC Cooling Main Power (kW)"] = np.inf
    elif failure == "missing_circuit":
        raw = raw.drop(columns="HVAC Cooling Main Power (kW)")
    else:
        raw.loc[23, "Time"] = raw.loc[22, "Time"]
    source = write_raw(tmp_path, raw)
    destination = tmp_path / "seed.csv"
    with pytest.raises(ValueError):
        prepare_ev_submeter_hub_seed(source, FIRST_ISSUE, destination)
    assert not destination.exists()


def test_timezone_mismatch_unaligned_cutoff_and_overwrite_are_rejected(tmp_path):
    source = write_raw(tmp_path, raw_history())
    destination = tmp_path / "seed.csv"
    with pytest.raises(ValueError, match="timezone"):
        prepare_ev_submeter_hub_seed(source, FIRST_ISSUE.tz_localize(None), destination)
    with pytest.raises(ValueError, match="aligned"):
        prepare_ev_submeter_hub_seed(source, FIRST_ISSUE + pd.Timedelta(minutes=1), destination)
    prepare_ev_submeter_hub_seed(source, FIRST_ISSUE, destination)
    before = destination.read_bytes()
    with pytest.raises(FileExistsError, match="new files"):
        prepare_ev_submeter_hub_seed(source, FIRST_ISSUE, destination)
    assert destination.read_bytes() == before


def test_mismatched_source_offset_is_rejected(tmp_path):
    raw = raw_history()
    raw["Time"] = raw["Time"].dt.tz_convert("Etc/GMT+4")
    source = write_raw(tmp_path, raw)
    with pytest.raises(ValueError, match="timezone offsets"):
        prepare_ev_submeter_hub_seed(source, FIRST_ISSUE, tmp_path / "seed.csv")
