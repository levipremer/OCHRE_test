"""Summarize metered HVAC and comfort outcomes for the matched 48-hour run."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples" / "single_home_hems"))
from examples.single_home_hems.single_home_hems.training.hvac import heating_setpoints_c


OUTPUT = Path(__file__).resolve().parent / "results"
DATA = OUTPUT / "data"


def summarize(frame: pd.DataFrame, price: pd.Series, er: pd.Series) -> dict:
    setpoint = heating_setpoints_c(frame.index)
    temperature = frame["indoor_temperature_c"].to_numpy(dtype=float)
    below = np.maximum(setpoint - 1.5 - temperature, 0)
    above = np.maximum(temperature - setpoint - 1.5, 0)
    excess = below + above
    hvac = frame["hvac_power_kw"]
    whole = frame["total_electric_power_kw"]
    return {
        "minutes": len(frame),
        "hvac_energy_kwh": float(hvac.sum() / 60),
        "hvac_cost_usd": float((hvac * price).sum() / 60),
        "hvac_peak_kw": float(hvac.max()),
        "resistance_energy_kwh": float(er.sum() / 60),
        "whole_home_energy_kwh": float(whole.sum() / 60),
        "whole_home_cost_usd": float((whole * price).sum() / 60),
        "whole_home_peak_kw": float(whole.max()),
        "indoor_min_c": float(temperature.min()),
        "indoor_max_c": float(temperature.max()),
        "within_scheduled_band_pct": float(100 * np.mean(excess <= 1e-9)),
        "outside_scheduled_band_hours": float(np.count_nonzero(excess > 1e-9) / 60),
        "degree_hours_below_band": float(below.sum() / 60),
        "degree_hours_above_band": float(above.sum() / 60),
        "max_deviation_beyond_band_c": float(excess.max()),
    }


def main() -> None:
    with (OUTPUT / "study_manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest["status"] != "complete":
        raise RuntimeError("Run is not complete")
    controlled = pd.read_csv(DATA / "home_01_standardized.csv", index_col=0, parse_dates=True)
    reference = pd.read_csv(DATA / "home_01_uncontrolled_standardized.csv", index_col=0, parse_dates=True)
    controls = pd.read_csv(DATA / "home_01_hems_controls.csv", index_col=0, parse_dates=True)
    controlled_raw = pd.read_csv(
        DATA / "raw" / "home_01.csv", index_col=0, parse_dates=True,
        usecols=["Time", "HVAC Heating ER Power (kW)"],
    )
    reference_raw = pd.read_csv(
        DATA / "raw" / "uncontrolled" / "home_01.csv", index_col=0, parse_dates=True,
        usecols=["Time", "HVAC Heating ER Power (kW)"],
    )
    if not controlled.index.equals(reference.index) or not controlled.index.equals(controls.index):
        raise ValueError("Matched timestamps differ")
    price = controls["electricity_price_usd_per_kwh"]
    result = {
        "evaluation_start_local": controlled.index[0].isoformat(),
        "evaluation_end_exclusive_local": (
            controlled.index[-1] + pd.Timedelta(minutes=1)
        ).isoformat(),
        "controlled": summarize(
            controlled, price,
            controlled_raw["HVAC Heating ER Power (kW)"].reindex(controlled.index),
        ),
        "native_reference": summarize(
            reference, price,
            reference_raw["HVAC Heating ER Power (kW)"].reindex(reference.index),
        ),
    }
    result["by_local_day"] = {}
    for day in sorted(set(controlled.index.date)):
        day_mask = controlled.index.date == day
        day_index = controlled.index[day_mask]
        result["by_local_day"][day.isoformat()] = {
            "controlled": summarize(
                controlled.loc[day_index], price.loc[day_index],
                controlled_raw["HVAC Heating ER Power (kW)"].reindex(day_index),
            ),
            "native_reference": summarize(
                reference.loc[day_index], price.loc[day_index],
                reference_raw["HVAC Heating ER Power (kW)"].reindex(day_index),
            ),
        }
    history_path = DATA / "hvac_optimization_solve_history.csv"
    if history_path.is_file():
        history = pd.read_csv(history_path)
        result["optimization"] = {
            "attempts": len(history),
            "successful": int(history["successful"].sum()),
            "statuses": history["status"].value_counts().to_dict(),
            "time_limits_hit": int(history["time_limit_hit"].sum()),
        }
    budget_path = DATA / "hvac_optimization_comfort_budget_history.csv"
    if budget_path.is_file():
        budget = pd.read_csv(budget_path)
        result["optimization"]["issues_with_budget_slack"] = int(
            (budget["degree_hour_slack_c_hours"] > 1e-6).sum()
        )
        result["optimization"]["issues_with_deviation_cap_slack"] = int(
            (budget["maximum_deviation_slack_c"] > 1e-6).sum()
        )
    (OUTPUT / "performance.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
