"""Compare aggregate and submeter-history RFs over a 20-hour Hub horizon.

This isolated extension reuses the frozen six-week resistance-excluded Home
Hub target and the original RF settings, chronological folds, and history
feature definitions. The augmented model adds past water-heater, outdoor
heat-pump-unit, and indoor-fan powers. All measurements must be completed
before the issue; no current/future circuit values enter either predictor.

Each final fit uses two, three, or four trailing weeks before week six. Four
one-day rolling folds in week five select leaf size from the existing (5, 15)
grid. The extended 80-step horizon leaves 17 complete origins per daily fold
and 593 in the historical seven-day test week. This is a forecast study on
saved plant data, not a new predictive-controller run or a deployed model.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path
import platform
import time

import joblib
import numpy as np
import pandas as pd

try:
    from . import evaluate_simulated_home_hub as original
except ImportError:
    import evaluate_simulated_home_hub as original


HERE = Path(__file__).resolve().parent
BASELINE = HERE.parent / "outputs/home_01/ev_home_hub_baseline_6wk_no_er"
DEFAULT_OUTPUT = BASELINE / "submeter_rf_20h_comparison_20261105"
INTERVAL = original.INTERVAL
DAY = original.DAY
WEEK = original.WEEK
HORIZON_STEPS = 80
HORIZON_HOURS = 20
TRAIN_WEEKS = original.TRAIN_WEEKS
CV_DAY_OFFSETS = original.CV_DAY_OFFSETS
LEAF_CANDIDATES = (5, 15)
METHODS = ("rf_aggregate_history", "rf_submeter_history")
POWER_LAGS = (1, 2, 4, 8, 96, 672)
ROLLING_STEPS = (4, 8, 96)
CIRCUIT_DEFINITIONS = {
    "water_heater_kw": ("Water Heating Electric Power (kW)",),
    "outdoor_heat_pump_kw": (
        "HVAC Heating Main Power (kW)", "HVAC Cooling Main Power (kW)",
    ),
    "indoor_fan_kw": (
        "HVAC Heating Fan Power (kW)", "HVAC Cooling Fan Power (kW)",
    ),
}
CIRCUIT_PREFIXES = {
    "water_heater_kw": "water_heater",
    "outdoor_heat_pump_kw": "outdoor_heat_pump",
    "indoor_fan_kw": "indoor_fan",
}
AUDIT_COLUMNS = (
    "HVAC Heating Electric Power (kW)", "HVAC Heating ER Power (kW)",
    "HVAC Cooling Electric Power (kW)", "DC House Hub Other Electric Power (kW)",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_submeter_power(raw_path: Path, data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Read complete observed circuits and audit them against the saved Hub."""
    power_columns = tuple(dict.fromkeys(
        column for sources in CIRCUIT_DEFINITIONS.values() for column in sources
    ))
    raw = pd.read_csv(raw_path, usecols=["Time", *power_columns, *AUDIT_COLUMNS],
                      parse_dates=["Time"]).set_index("Time")
    if raw.index.tz is not None:
        raw.index = raw.index.tz_localize(None)
    if raw.index.has_duplicates or not raw.index.is_monotonic_increasing:
        raise ValueError("Raw OCHRE circuit timestamps must be unique and increasing")
    start, end = data.index[0], data.index[-1] + INTERVAL
    raw = raw.loc[(raw.index >= start) & (raw.index < end)]
    expected = pd.date_range(start, end, freq="min", inclusive="left")
    if not raw.index.equals(expected):
        raise ValueError("Submeter inputs require every minute of each target block")
    raw = raw.apply(pd.to_numeric, errors="raise")
    if not np.isfinite(raw.to_numpy(float)).all() or raw.lt(-1e-8).any().any():
        raise ValueError("Submeter powers must be finite and nonnegative")
    heating_error = (
        raw["HVAC Heating Electric Power (kW)"]
        - raw["HVAC Heating Main Power (kW)"]
        - raw["HVAC Heating Fan Power (kW)"]
        - raw["HVAC Heating ER Power (kW)"]
    )
    cooling_error = (
        raw["HVAC Cooling Electric Power (kW)"]
        - raw["HVAC Cooling Main Power (kW)"]
        - raw["HVAC Cooling Fan Power (kW)"]
    )
    if heating_error.abs().max() > 1e-8 or cooling_error.abs().max() > 1e-8:
        raise ValueError("OCHRE HVAC main/fan/resistance decomposition changed")
    minute = pd.DataFrame({
        name: raw.loc[:, list(columns)].sum(axis=1)
        for name, columns in CIRCUIT_DEFINITIONS.items()
    }, index=raw.index)
    blocks = minute.resample("15min", label="left", closed="left").mean()
    blocks.index.name = "timestamp"
    count = minute.resample("15min", label="left", closed="left").count()
    if not blocks.index.equals(data.index) or not count.eq(15).all().all():
        raise ValueError("Submeter blocks differ from the complete Hub grid")
    expected_hub = (
        minute.sum(axis=1) + raw["DC House Hub Other Electric Power (kW)"]
    ).resample("15min", label="left", closed="left").mean()
    error = expected_hub.to_numpy(float) - data["hub_non_ev_power_kw"].to_numpy(float)
    if not np.allclose(error, 0.0, rtol=0, atol=1e-8):
        raise ValueError("Submeter circuits plus Hub-other do not equal corrected Hub target")
    return blocks, {
        "raw_minutes": len(raw), "complete_15min_blocks": len(blocks),
        "aggregate_identity_max_abs_error_kw": float(np.abs(error).max()),
        "heating_decomposition_max_abs_error_kw": float(heating_error.abs().max()),
        "cooling_decomposition_max_abs_error_kw": float(cooling_error.abs().max()),
        "outdoor_unit_standby_note": (
            "Outdoor unit is Heating Main + Cooling Main; this includes the "
            "saved cooling unit's 0.05 kW standby/crankcase power. It excludes "
            "indoor fan and HVAC electric resistance."
        ),
    }


def power_history(series: pd.Series, prefix: str) -> pd.DataFrame:
    """Replicate the established aggregate-power history pattern for one meter."""
    result = pd.DataFrame(index=series.index)
    for lag in POWER_LAGS:
        values = series.shift(lag)
        result[f"{prefix}_lag_{lag}_kw"] = values.fillna(0.0)
        result[f"{prefix}_lag_{lag}_available"] = values.notna().astype(int)
    previous = series.shift(1)
    for steps in ROLLING_STEPS:
        rolling = previous.rolling(steps, min_periods=1)
        result[f"{prefix}_mean_prev_{steps}_kw"] = rolling.mean().fillna(0.0)
        result[f"{prefix}_max_prev_{steps}_kw"] = rolling.max().fillna(0.0)
        result[f"{prefix}_count_prev_{steps}"] = rolling.count().astype(int)
    change = previous - series.shift(5)
    result[f"{prefix}_slope_1h_kw"] = change.fillna(0.0)
    result[f"{prefix}_slope_1h_available"] = change.notna().astype(int)
    return result


@dataclass(frozen=True)
class StudyRows:
    x: pd.DataFrame
    y: np.ndarray
    origin: pd.DatetimeIndex
    target: pd.DatetimeIndex
    lead: np.ndarray
    feature_sets: dict[str, tuple[str, ...]]
    horizon_steps: int = HORIZON_STEPS

    def mask_fit(self, start: pd.Timestamp, cutoff: pd.Timestamp) -> np.ndarray:
        return np.asarray(
            (self.origin >= start)
            & (self.origin + self.horizon_steps * INTERVAL <= cutoff)
            & ((self.origin.hour * 4 + self.origin.minute // 15) % 4 == 0)
            & np.isfinite(self.y), dtype=bool,
        )

    def mask_evaluation(self, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
        return np.asarray(
            (self.origin >= start)
            & (self.origin + self.horizon_steps * INTERVAL <= end)
            & np.isfinite(self.y), dtype=bool,
        )


def make_supervised(data: pd.DataFrame, circuits: pd.DataFrame,
                    horizon_steps: int = HORIZON_STEPS) -> StudyRows:
    """Create paired feature sets with identical origins, targets, and labels."""
    if (not circuits.index.equals(data.index)
            or tuple(circuits.columns) != tuple(CIRCUIT_DEFINITIONS)):
        raise ValueError("Circuit inputs must match the aggregate grid and three-meter schema")
    if not np.isfinite(circuits.to_numpy(float)).all():
        raise ValueError("Circuit observations contain missing or nonfinite power")
    if horizon_steps < 1 or horizon_steps >= len(data):
        raise ValueError("Prediction horizon must leave at least one observed history row")
    index = data.index
    values = data["hub_non_ev_power_kw"].to_numpy(float)
    aggregate = power_history(data["hub_non_ev_power_kw"], "load")
    if "ambient_temperature_c" in data:
        ambient = data["ambient_temperature_c"].shift(1)
        aggregate["ambient_lag_1_c"] = ambient.fillna(0.0)
        aggregate["ambient_lag_1_available"] = ambient.notna().astype(int)
    extra = pd.concat([
        power_history(circuits[name], CIRCUIT_PREFIXES[name])
        for name in CIRCUIT_DEFINITIONS
    ], axis=1)
    history = pd.concat((aggregate, extra), axis=1)
    origin_positions = np.repeat(np.arange(1, len(data) - horizon_steps + 1), horizon_steps)
    leads = np.tile(np.arange(1, horizon_steps + 1), len(origin_positions) // horizon_steps)
    targets = origin_positions + leads - 1
    origin_times, target_times = index.take(origin_positions), index.take(targets)
    x = history.iloc[origin_positions].reset_index(drop=True)
    x["lead_step"] = leads
    clock = 2 * np.pi * (target_times.hour * 60 + target_times.minute) / 1440
    week = 2 * np.pi * target_times.dayofweek / 7
    x["target_clock_sin"], x["target_clock_cos"] = np.sin(clock), np.cos(clock)
    x["target_week_sin"], x["target_week_cos"] = np.sin(week), np.cos(week)
    x["target_is_weekend"] = (target_times.dayofweek >= 5).astype(int)
    aggregate_columns = original.CALENDAR_COLUMNS + tuple(aggregate)
    augmented_columns = aggregate_columns + tuple(extra)
    if not np.isfinite(x.loc[:, list(augmented_columns)].to_numpy(float)).all():
        raise ValueError("RF features contain nonfinite values")
    return StudyRows(x, values[targets], origin_times, target_times, leads,
                     {METHODS[0]: aggregate_columns, METHODS[1]: augmented_columns},
                     horizon_steps)


def prediction_rows(rows: StudyRows, mask: np.ndarray, forecast: np.ndarray,
                    method: str, weeks: int) -> pd.DataFrame:
    if forecast.shape != (int(mask.sum()),) or not np.isfinite(forecast).all():
        raise ValueError("RF returned an invalid prediction array")
    return pd.DataFrame({
        "method": method, "train_weeks": weeks, "split": "test",
        "origin": rows.origin[mask], "target": rows.target[mask], "lead": rows.lead[mask],
        "forecast_kw": np.maximum(forecast, 0.0), "actual_kw": rows.y[mask],
    })


def score_predictions(predictions: pd.DataFrame, threshold_kw: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Preserve metric definitions and extend horizon summaries through 20 h."""
    summary, lead = original.score_predictions(predictions, threshold_kw)
    extra = []
    for (method, weeks, split), frame in predictions.groupby(
        ["method", "train_weeks", "split"], sort=False
    ):
        for hours in (12, 16, 20):
            for scope, selected in (
                ("through_horizon", frame.loc[frame["lead"].le(4 * hours)]),
                ("at_horizon", frame.loc[frame["lead"].eq(4 * hours)]),
            ):
                extra.append({"method": method, "train_weeks": weeks, "split": split,
                              "horizon_hours": hours, "scope": scope,
                              **original._score(selected, threshold_kw)})
    return pd.concat((summary, pd.DataFrame(extra)), ignore_index=True), lead


def save_predictions(frame: pd.DataFrame, path: Path) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        frame.to_csv(stream, index=False)


def run(meter_path: Path, weather_path: Path, raw_path: Path, output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new empty output directory: {output}")
    data = original.read_baseline(meter_path, weather_15min_path=weather_path)
    circuits, circuit_audit = read_submeter_power(raw_path, data)
    rows = make_supervised(data, circuits)
    start = data.index[0]
    validation_start, test_start, test_end = start + 4 * WEEK, start + 5 * WEEK, start + 6 * WEEK
    test_mask = rows.mask_evaluation(test_start, test_end)
    if rows.origin[test_mask].nunique() != 593 or int(test_mask.sum()) != 593 * HORIZON_STEPS:
        raise AssertionError("The 20-hour held-out grid must contain 593 complete issues")
    threshold = float(data.loc[data.index < start + 20 * DAY, "hub_non_ev_power_kw"].quantile(0.95))
    output.mkdir(parents=True, exist_ok=True)
    (output / "models").mkdir()
    evaluator_source = Path(__file__).resolve()
    evaluator_snapshot = output / evaluator_source.name
    evaluator_snapshot.write_bytes(evaluator_source.read_bytes())
    circuits.to_csv(output / "submeter_power_15min.csv")
    cv_records, primary, sensitivity, importance = [], [], [], []
    selections, fit_records = {}, []
    started = time.monotonic()
    for weeks in TRAIN_WEEKS:
        for method in METHODS:
            columns = list(rows.feature_sets[method])
            x_test = rows.x.loc[test_mask, columns]
            means = {}
            for leaf in LEAF_CANDIDATES:
                fold_scores = []
                for day in CV_DAY_OFFSETS:
                    cutoff = validation_start + day * DAY
                    fit = rows.mask_fit(cutoff - weeks * WEEK, cutoff)
                    valid = rows.mask_evaluation(cutoff, cutoff + DAY)
                    if rows.origin[valid].nunique() != 17 or int(valid.sum()) != 17 * HORIZON_STEPS:
                        raise AssertionError("Each daily 20-hour CV fold must have 17 complete issues")
                    model = original._model("rf_history", {"min_samples_leaf": leaf})
                    model.fit(rows.x.loc[fit, columns], rows.y[fit])
                    forecast = np.maximum(np.asarray(model.predict(rows.x.loc[valid, columns])), 0.0)
                    mae = float(np.abs(forecast - rows.y[valid]).mean())
                    fold_scores.append(mae)
                    cv_records.append({
                        "method": method, "train_weeks": weeks, "min_samples_leaf": leaf,
                        "fold_start": cutoff, "fold_end": cutoff + DAY,
                        "fit_start": cutoff - weeks * WEEK, "fit_end_exclusive": cutoff,
                        "fit_pairs": int(fit.sum()), "fit_origins": rows.origin[fit].nunique(),
                        "fit_last_target_end": rows.target[fit].max() + INTERVAL,
                        "validation_pairs": int(valid.sum()), "validation_origins": 17,
                        "mae_kw": mae,
                    })
                means[leaf] = float(np.mean(fold_scores))
                print(f"{weeks}w {method} leaf={leaf}: mean CV MAE {means[leaf]:.6f} kW", flush=True)
            selected_leaf = min(LEAF_CANDIDATES, key=lambda leaf: means[leaf])
            selections[f"{weeks}w:{method}"] = {
                "min_samples_leaf": selected_leaf, "cv_mae_kw": means[selected_leaf],
                "candidate_cv_mae_kw": means,
            }
            fit = rows.mask_fit(test_start - weeks * WEEK, test_start)
            for leaf in sorted({selected_leaf, 15}):
                model = original._model("rf_history", {"min_samples_leaf": leaf})
                model.fit(rows.x.loc[fit, columns], rows.y[fit])
                forecast = np.asarray(model.predict(x_test), dtype=float)
                predicted = prediction_rows(rows, test_mask, forecast, method, weeks)
                model_file = output / "models" / f"{method}_{weeks}w_leaf{leaf}.joblib"
                joblib.dump({"model": model, "feature_columns": columns,
                             "horizon_steps": HORIZON_STEPS,
                             "train_weeks": weeks, "min_samples_leaf": leaf,
                             "fit_start": str(test_start - weeks * WEEK),
                             "fit_end_exclusive": str(test_start)}, model_file)
                fit_records.append({
                    "method": method, "train_weeks": weeks, "min_samples_leaf": leaf,
                    "fit_start": str(test_start - weeks * WEEK),
                    "fit_end_exclusive": str(test_start), "fit_pairs": int(fit.sum()),
                    "fit_origins": int(rows.origin[fit].nunique()),
                    "last_fit_target_end": str(rows.target[fit].max() + INTERVAL),
                    "model_file": str(model_file.resolve()),
                    "model_sha256": sha256(model_file),
                })
                if leaf == selected_leaf:
                    primary.append(predicted)
                    importance.extend({"method": method, "train_weeks": weeks,
                                       "feature": name, "impurity_importance": float(value)}
                                      for name, value in zip(columns, model.feature_importances_))
                if leaf == 15:
                    sensitivity.append(predicted)
            print(f"{weeks}w {method}: selected leaf={selected_leaf}; test MAE "
                  f"{np.abs(primary[-1].forecast_kw-primary[-1].actual_kw).mean():.6f} kW", flush=True)
            pd.DataFrame(cv_records).to_csv(output / "rolling_cv.csv", index=False)
    predictions = pd.concat(primary, ignore_index=True)
    fixed = pd.concat(sensitivity, ignore_index=True)
    for prefix, frame in (("", predictions), ("fixed_leaf15_", fixed)):
        save_predictions(frame, output / f"{prefix}predictions.csv.gz")
        scores, leads = score_predictions(frame, threshold)
        scores.to_csv(output / f"{prefix}horizon_scores.csv", index=False)
        leads.to_csv(output / f"{prefix}lead_scores.csv", index=False)
        original.score_by_issue_day(frame, threshold).to_csv(
            output / f"{prefix}issue_day_scores.csv", index=False)
        twenty = scores.loc[scores["horizon_hours"].eq(20) & scores["scope"].eq("through_horizon")]
        twenty.to_csv(output / f"{prefix}twenty_hour_scores.csv", index=False)
        compared = twenty.pivot(index="train_weeks", columns="method", values="mae_kw")
        compared = compared.rename(columns={METHODS[0]: "aggregate_mae_kw", METHODS[1]: "submeter_mae_kw"})
        compared["mae_improvement_kw"] = compared.aggregate_mae_kw - compared.submeter_mae_kw
        compared["mae_improvement_percent"] = 100 * compared.mae_improvement_kw / compared.aggregate_mae_kw
        compared.to_csv(output / f"{prefix}twenty_hour_comparison.csv")
    pd.DataFrame(importance).to_csv(output / "feature_importance.csv", index=False)
    chosen_windows = {
        method: min(TRAIN_WEEKS, key=lambda weeks: selections[f"{weeks}w:{method}"]["cv_mae_kw"])
        for method in METHODS
    }
    metadata = {
        "study": "aggregate vs aggregate-plus-circuit-history RF, 20-hour forecast",
        "target": "15-minute non-EV Home Hub power = corrected Hub meter minus signed EV meter; HVAC ER excluded",
        "input_files": {"meter": str(meter_path.resolve()), "weather": str(weather_path.resolve()),
                        "raw_circuits": str(raw_path.resolve())},
        "input_sha256": {"meter": sha256(meter_path), "weather": sha256(weather_path),
                         "raw_circuits": sha256(raw_path)},
        "evaluator_source": str(evaluator_source),
        "evaluator_snapshot": str(evaluator_snapshot.resolve()),
        "evaluator_sha256": sha256(evaluator_snapshot),
        "original_rf_helper_source": str(Path(original.__file__).resolve()),
        "original_rf_helper_sha256": sha256(Path(original.__file__)),
        "submeter_export_sha256": sha256(output / "submeter_power_15min.csv"),
        "circuit_definitions": CIRCUIT_DEFINITIONS, "circuit_audit": circuit_audit,
        "feature_columns": {name: list(columns) for name, columns in rows.feature_sets.items()},
        "feature_counts": {name: len(columns) for name, columns in rows.feature_sets.items()},
        "power_history": {"lag_15min_steps": POWER_LAGS, "rolling_mean_max_count_steps": ROLLING_STEPS,
                          "change": "latest completed 15-minute mean minus the mean one hour earlier",
                          "availability": "same flags/counts as original aggregate-history RF"},
        "nonpower_features": "latest completed 15-minute ambient temperature, target clock/day/weekend, forecast lead",
        "causality": "All circuit, aggregate, and weather features use completed readings strictly before the forecast issue.",
        "historical_context": "As in the original benchmark, feature lookbacks can precede the trailing target-training window.",
        "baseline_start": str(start), "validation_start": str(validation_start),
        "test_start": str(test_start), "test_end_exclusive": str(test_end),
        "test_window_note": "Original frozen seven-day week-six forecast benchmark retained; no new predictive-controller simulation.",
        "interval_minutes": 15, "horizon_steps": HORIZON_STEPS, "horizon_hours": HORIZON_HOURS,
        "training_weeks": TRAIN_WEEKS, "cv_fold_day_offsets_in_week5": CV_DAY_OFFSETS,
        "cv_issues_per_fold": 17, "cv_unique_issue_count_per_window": 68,
        "cv_issue_clock_span": "00:00 through 04:00 in each one-day fold; horizons remain within that day",
        "cv_rule": "Original four rolling one-day folds; full 20-hour label purge; lowest mean fold MAE selects leaf size",
        "fit_origin_stride_minutes": 60, "test_origin_stride_minutes": 15,
        "test_origins": 593, "test_pairs_per_method_window": 593 * HORIZON_STEPS,
        "first_test_issue": str(rows.origin[test_mask].min()),
        "last_test_issue": str(rows.origin[test_mask].max()),
        "last_test_target_end": str(rows.target[test_mask].max() + INTERVAL),
        "rf_settings": {"n_estimators": 80, "max_features": 0.8, "max_samples": 0.8,
                        "bootstrap": True, "n_jobs": 4, "random_state": original.SEED},
        "leaf_candidates": LEAF_CANDIDATES, "selected_validation_params": selections,
        "validation_selected_training_weeks": chosen_windows,
        "fixed_leaf15_sensitivity": "Same paired feature sets/window/grids with the deployed RF's existing leaf size15, separate from CV-selected primary scores",
        "peak_threshold_kw": threshold,
        "peak_threshold_rule": "95th percentile of first 20 baseline days, unchanged target and prevalidation rule",
        "final_fit_records": fit_records,
        "runtime_integration": "None; this experiment only predicts mean power and does not replace controller artifacts",
        "versions": {"python": platform.python_version(), **{
            name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "joblib")}},
        "wall_seconds": time.monotonic() - started,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "wall_seconds": metadata["wall_seconds"],
                      "validation_selected_training_weeks": chosen_windows}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meter", type=Path, default=BASELINE / "data/home_hub_meter_15min.csv")
    parser.add_argument("--weather", type=Path, default=BASELINE / "data/realized_weather_15min.csv")
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    raw = args.raw
    if raw is None:
        metadata = json.loads((BASELINE / "baseline_metadata.json").read_text())
        raw = Path(metadata["source_raw"])
    run(args.meter, args.weather, raw, args.output_dir)


if __name__ == "__main__":
    main()
