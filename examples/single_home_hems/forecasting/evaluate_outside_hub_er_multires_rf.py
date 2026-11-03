"""Compare multiresolution aggregate-load RFs for EV-only Outside-Hub power.

The target includes HVAC electric resistance outside the corrected Home Hub.
Load history uses only originally observed aggregate meter minutes. Future
weather is either a causal persistence forecast or an explicitly noncausal
perfect-weather upper bound. Existing forecast/controller outputs are untouched.
"""

from __future__ import annotations

import argparse
import gzip
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .evaluate_outside_hub import CV_DAY_OFFSETS, DAY, SIMULATION_START, supervised_rows
    from .evaluate_outside_hub_with_er import (
        CORRECTED_BASELINE_DIR, DEFAULT_OUTPUT as CORE_DIR, _sha256,
        load_composite_target,
    )
    from .evaluate_outside_hub_with_er_direct_qcp import _saved_weather_rf
    from .evaluate_outside_hub_with_er_rf_qcp import (
        DEFAULT_OUTPUT as PAST_AMBIENT_DIR, LEAVES, _common_origins, _saved_rf,
    )
    from .evaluate_simulated_home_hub import (
        HORIZON_STEPS, TRAIN_WEEKS, WEEK, score_by_issue_day, score_predictions,
    )
    from .outside_hub_candidate_models import make_candidate
    from .outside_hub_er_multires_load import load_feature_matrix, load_minute_observed
    from .outside_hub_er_weather_features import weather_feature_matrices
except ImportError:  # direct script invocation from repository root
    from evaluate_outside_hub import CV_DAY_OFFSETS, DAY, SIMULATION_START, supervised_rows
    from evaluate_outside_hub_with_er import (
        CORRECTED_BASELINE_DIR, DEFAULT_OUTPUT as CORE_DIR, _sha256,
        load_composite_target,
    )
    from evaluate_outside_hub_with_er_direct_qcp import _saved_weather_rf
    from evaluate_outside_hub_with_er_rf_qcp import (
        DEFAULT_OUTPUT as PAST_AMBIENT_DIR, LEAVES, _common_origins, _saved_rf,
    )
    from evaluate_simulated_home_hub import (
        HORIZON_STEPS, TRAIN_WEEKS, WEEK, score_by_issue_day, score_predictions,
    )
    from outside_hub_candidate_models import make_candidate
    from outside_hub_er_multires_load import load_feature_matrix, load_minute_observed
    from outside_hub_er_weather_features import weather_feature_matrices


OUTPUT = CORE_DIR / "supplemental/rf_multires_weather"
WEATHER_MINUTE = CORRECTED_BASELINE_DIR / "data/realized_weather_minute.csv"
WEATHER_15MIN = CORRECTED_BASELINE_DIR / "data/realized_weather_15min.csv"
METHODS = (
    "rf_multires_load",
    "rf_multires_weather_persistence",
    "rf_multires_weather_oracle",
)
OLD_LAGS = tuple(
    column for lag in (1, 2, 4, 8, 96, 672)
    for column in (f"load_lag_{lag}_kw", f"load_lag_{lag}_available")
)


def _feature_arrays(rows, minute_observed: pd.Series) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    unique_origins = pd.DatetimeIndex(rows.origin.unique())
    minute_history = load_feature_matrix(minute_observed, unique_origins)
    if len(minute_history) != len(unique_origins):
        raise ValueError("Minute history must have one row per forecast origin")
    if not minute_history.index.equals(unique_origins):
        raise ValueError("Minute history origin index is misaligned")
    repeated_history = minute_history.loc[rows.origin].reset_index(drop=True)
    retained = rows.x.drop(columns=list(OLD_LAGS)).reset_index(drop=True)
    load = pd.concat((retained, repeated_history), axis=1)
    if load.columns.has_duplicates:
        raise ValueError("New load-history columns overlap retained RF columns")
    weather = weather_feature_matrices(rows.origin, rows.target, WEATHER_MINUTE)
    if set(weather) != {"past_weather_persistence", "past_weather_oracle"}:
        raise ValueError("Weather feature builder omitted an approved weather variant")
    frames = {
        METHODS[0]: load,
        METHODS[1]: pd.concat((load, weather["past_weather_persistence"]), axis=1),
        METHODS[2]: pd.concat((load, weather["past_weather_oracle"]), axis=1),
    }
    columns = {name: list(frame.columns) for name, frame in frames.items()}
    arrays: dict[str, np.ndarray] = {}
    for name, frame in frames.items():
        if len(frame) != len(rows.y) or frame.columns.has_duplicates:
            raise ValueError(f"{name} features have wrong shape or duplicate columns")
        values = np.asarray(frame.to_numpy(dtype=np.float32), dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"{name} features include nonfinite values")
        arrays[name] = np.ascontiguousarray(values)
    if columns[METHODS[1]] != columns[METHODS[2]]:
        raise ValueError("Causal and oracle weather variants must have identical columns")
    return arrays, columns


def _forecast_rows(rows, mask: np.ndarray, forecast: np.ndarray,
                   method: str, weeks: int) -> pd.DataFrame:
    if len(forecast) != int(mask.sum()) or not np.isfinite(forecast).all():
        raise ValueError("RF forecast count or finite-value check failed")
    return pd.DataFrame({
        "method": method, "train_weeks": weeks, "split": "test",
        "origin": rows.origin[mask], "target": rows.target[mask],
        "lead": rows.lead[mask], "forecast_kw": forecast,
        "actual_kw": rows.y[mask],
    })


def _predict(model: Any, matrix: np.ndarray) -> np.ndarray:
    forecast = np.asarray(model.predict(matrix), dtype=float)
    if not np.isfinite(forecast).all():
        raise ValueError("RF produced a nonfinite forecast")
    return np.maximum(forecast, 0.0)


def _report(output_dir: Path, scores: pd.DataFrame,
            selected: dict[str, dict[str, Any]],
            columns: dict[str, list[str]], threshold_kw: float,
            load_audit: dict[str, Any]) -> None:
    eight = scores.loc[
        scores["scope"].eq("through_horizon") & scores["horizon_hours"].eq(8)
    ].sort_values(["train_weeks", "mae_kw"])
    lines = [
        "# EV-only Outside-Hub plus resistance: multiresolution RF history",
        "",
        "Target: measured aggregate total-home minus corrected Home Hub, including",
        "OCHRE HVAC electric resistance outside the Hub. Training labels remain",
        "originally observed 15-minute aggregate bins. The complete replay is",
        "used only for validation and test scoring.",
        "",
        "Power history: 15 last completed one-minute values, seven complete",
        "15-minute block means extending to two hours, and 22 complete hourly",
        "block means extending to 24 hours. Each has an availability flag.",
        "The original 1-, 2-, and 24-hour rolling summaries, slope, and timing",
        "features remain. The weather variants add 24 past completed hourly",
        "temperature and relative-humidity means, plus target-time temperature",
        "and humidity. Causal persistence holds the latest completed hourly mean",
        "through the eight-hour horizon. Oracle uses the future realized TMY",
        "weather and is an optimistic upper bound, not an issued forecast.",
        "The controller-facing display name for `rf_multires_weather_persistence`",
        "is **Multiresolution RF + weather persistence**; the saved method ID",
        "remains unchanged for reproducibility.",
        "",
        "Training: 2, 3, and 4 weeks; four chronological week-5 validation",
        "days for RF leaf choice; week 6 for the test. Saved historical RFs are",
        "compared on the same 640 issues shared with the RF-to-p90 comparator.",
        f"A peak is a 15-minute target at or above {threshold_kw:.6f} kW,",
        "the fixed weeks-1–4 originally observed p95 threshold.",
        "",
        "| Training | Method | Features | 8 h MAE (kW) | Bias (kW) | Underforecast | Peak recall | Peak shortfall (kW) |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in eight.itertuples():
        count = len(columns[row.method]) if row.method in columns else (31 if row.method == "rf_history_weather_lag" else 29)
        lines.append(
            f"| {row.train_weeks} weeks | {row.method} | {count} | "
            f"{row.mae_kw:.3f} | {row.bias_kw:+.3f} | "
            f"{row.underforecast_rate:.3f} | {row.peak_recall:.3f} | "
            f"{row.actual_peak_mean_shortfall_kw:.3f} |"
        )
    lines += [
        "", "Feature counts include power, weather, availability, calendar, and lead columns.",
        f"Minute aggregate source: {load_audit.get('source_path', 'see metadata')}.",
        "", "[Validation folds](rolling_cv.csv) · [All test scores](full_horizon_scores.csv) ·",
        "[Matched test scores](matched_horizon_scores.csv) · [Lead scores](matched_lead_scores.csv) ·",
        "[Feature definitions and provenance](metadata.json)", "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run(*, output_dir: Path = OUTPUT) -> None:
    output_dir = output_dir.expanduser().resolve()
    truth, observed, target_audit = load_composite_target()
    rows, observed_labels = supervised_rows(observed, truth)
    minute_observed, load_audit = load_minute_observed(truth, observed)
    x, columns = _feature_arrays(rows, minute_observed)
    test_start = SIMULATION_START + 5 * WEEK
    test_end = SIMULATION_START + 6 * WEEK
    validation_start = SIMULATION_START + 4 * WEEK
    test = rows.mask_evaluation(test_start, test_end)
    if int(test.sum()) != 641 * HORIZON_STEPS:
        raise ValueError("Test grid differs from the saved Outside-Hub benchmark")
    baseline, baseline_provenance = _saved_rf(CORE_DIR, rows, test, target_audit)
    past_ambient, ambient_provenance = _saved_weather_rf(
        PAST_AMBIENT_DIR, rows, test, target_audit, _sha256(WEATHER_15MIN)
    )
    common_origins, matched_provenance = _common_origins(CORE_DIR)
    if not common_origins.isin(pd.DatetimeIndex(rows.origin[test].unique())).all():
        raise ValueError("Hybrid-matched issue grid is not in the direct-model grid")

    output_dir.mkdir(parents=True, exist_ok=True)
    observed_target = np.isfinite(observed_labels)
    cv_records: list[dict[str, Any]] = []
    selected: dict[str, dict[str, Any]] = {}
    forecasts = [baseline, past_ambient]
    for weeks in TRAIN_WEEKS:
        fold_masks = []
        for offset in CV_DAY_OFFSETS:
            fold_start = validation_start + offset * DAY
            fold_end = fold_start + DAY
            train_start = fold_start - weeks * WEEK
            fit = rows.mask_fit(train_start, fold_start) & observed_target
            valid = rows.mask_evaluation(fold_start, fold_end)
            if not fit.any() or not valid.any():
                raise ValueError("Multiresolution RF has an empty fit or validation fold")
            fold_masks.append((offset, fold_start, fold_end, train_start, fit, valid))
        final_start = test_start - weeks * WEEK
        final_fit = rows.mask_fit(final_start, test_start) & observed_target
        if not final_fit.any():
            raise ValueError("Multiresolution RF final fit has no observed labels")

        for method in METHODS:
            mean_errors = []
            for leaf in LEAVES:
                fold_errors = []
                for offset, fold_start, fold_end, train_start, fit, valid in fold_masks:
                    model = make_candidate("rf_history", {"min_samples_leaf": leaf})
                    model.fit(x[method][fit], observed_labels[fit])
                    forecast = _predict(model, x[method][valid])
                    error = float(np.mean(np.abs(forecast - rows.y[valid])))
                    fold_errors.append(error)
                    cv_records.append({
                        "train_weeks": weeks, "method": method, "leaf": leaf,
                        "fold_day_offset": offset, "fold_start": fold_start,
                        "fold_end": fold_end, "fit_start": train_start,
                        "fit_end_exclusive": fold_start, "fit_pairs": int(fit.sum()),
                        "validation_pairs": int(valid.sum()), "mae_kw": error,
                    })
                    pd.DataFrame(cv_records).to_csv(output_dir / "rolling_cv.csv", index=False)
                    print(f"{weeks}w {method} leaf={leaf} fold={offset}: {error:.3f} kW", flush=True)
                mean_errors.append(float(np.mean(fold_errors)))
            best = int(np.argmin(mean_errors))
            chosen_leaf = LEAVES[best]
            selected[f"{weeks}w:{method}"] = {
                "min_samples_leaf": chosen_leaf,
                "cv_mae_kw": mean_errors[best],
                "cv_leaf_mae_kw": dict(zip(map(str, LEAVES), mean_errors)),
            }
            model = make_candidate("rf_history", {"min_samples_leaf": chosen_leaf})
            model.fit(x[method][final_fit], observed_labels[final_fit])
            forecast = _predict(model, x[method][test])
            forecasts.append(_forecast_rows(rows, test, forecast, method, weeks))
            print(f"{weeks}w {method}: test forecast complete", flush=True)

    all_predictions = pd.concat(forecasts, ignore_index=True)
    all_predictions = all_predictions.sort_values(["train_weeks", "method", "origin", "lead"])
    matched = all_predictions.loc[all_predictions["origin"].isin(common_origins)]
    expected_groups = len(TRAIN_WEEKS) * (len(METHODS) + 2)
    for label, frame, expected_origins in (
        ("full", all_predictions, 641), ("matched", matched, 640)
    ):
        coverage = frame.groupby(["train_weeks", "method"])["origin"].nunique()
        if len(coverage) != expected_groups or not coverage.eq(expected_origins).all():
            raise ValueError(f"{label} RF methods lack matched complete origins")
    threshold_kw = float(json.loads((CORE_DIR / "metadata.json").read_text())[
        "peak_threshold_kw"
    ])
    full_scores, _ = score_predictions(all_predictions, threshold_kw)
    matched_scores, matched_leads = score_predictions(matched, threshold_kw)
    with gzip.open(output_dir / "predictions.csv.gz", "wt", encoding="utf-8") as stream:
        all_predictions.to_csv(stream, index=False)
    pd.DataFrame(cv_records).to_csv(output_dir / "rolling_cv.csv", index=False)
    full_scores.to_csv(output_dir / "full_horizon_scores.csv", index=False)
    matched_scores.to_csv(output_dir / "matched_horizon_scores.csv", index=False)
    matched_leads.to_csv(output_dir / "matched_lead_scores.csv", index=False)
    score_by_issue_day(matched, threshold_kw).to_csv(
        output_dir / "matched_issue_day_scores.csv", index=False
    )
    metadata = {
        "target": target_audit["target"],
        "target_observation_mask": "originally observed Outside-Hub source bins only",
        "load_feature_rule": "15 completed minute values, 7 complete 15-minute block means, 22 complete hourly block means; each with observed-only availability",
        "power_rolling_and_slope_rule": "unchanged from original 15-minute RF builder",
        "past_weather_rule": "24 trailing complete hourly temperature and humidity means before each issue",
        "future_weather_persistence_rule": "latest complete pre-issue hour mean held throughout horizon",
        "future_weather_oracle_rule": "future realized TMY weather in each target interval; noncausal upper bound",
        "rf_spec": "80 trees, 0.8 max_features, 0.8 max_samples, bootstrap, seed 42",
        "rf_leaf_grid": LEAVES,
        "training_weeks": TRAIN_WEEKS,
        "cv_fold_day_offsets_in_week5": CV_DAY_OFFSETS,
        "test_interval": [test_start.isoformat(), test_end.isoformat()],
        "full_test_origins": 641, "matched_test_origins": 640,
        "peak_threshold_kw": threshold_kw,
        "selected_validation_params": selected,
        "feature_columns": columns,
        "load_feature_audit": load_audit,
        "target_audit": target_audit,
        "realized_weather_minute_sha256": _sha256(WEATHER_MINUTE),
        "realized_weather_15min_sha256": _sha256(WEATHER_15MIN),
        **baseline_provenance, **ambient_provenance, **matched_provenance,
        "source_module_sha256": _sha256(Path(__file__)),
        "load_module_sha256": _sha256(Path(__file__).with_name("outside_hub_er_multires_load.py")),
        "weather_module_sha256": _sha256(Path(__file__).with_name("outside_hub_er_weather_features.py")),
        "package_versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scikit-learn")
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8"
    )
    _report(output_dir, matched_scores, selected, columns, threshold_kw, load_audit)
    print(f"Saved multiresolution RF comparison to {output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
