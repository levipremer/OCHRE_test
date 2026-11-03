"""Supplemental RF feature comparison for the EV-only Outside-Hub-plus-ER target.

The saved RF history forecast is the reference. New RFs use the same aggregate
target, originally observed fit labels, rolling week-five folds, held-out week
six, estimator, and leaf-size grid. One receives only the latest completed
ambient reading; another receives the *existing approved* Home Hub-minus-EV
QCP heating feature from completed corrected Hub meter and weather hours; a
fourth RF receives both to isolate QCP's gain beyond the raw past temperature.
QCP is a cross-meter heating proxy here, not an estimate of electric resistance.

Only the 640 week-six issues shared with the existing RF-to-p90 comparator are
used for the headline scores. All 641 direct-model issues are also retained.
This study does not alter the EV controller or the earlier benchmark files.
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
    from .evaluate_outside_hub import (
        CV_DAY_OFFSETS, DAY, SIMULATION_START, supervised_rows,
    )
    from .evaluate_outside_hub_with_er import (
        CORRECTED_BASELINE_DIR, DEFAULT_OUTPUT as CORE_DIR, _sha256,
        load_composite_target,
    )
    from .evaluate_simulated_home_hub import (
        HORIZON_STEPS, INTERVAL, QCP_COLUMNS, TRAIN_WEEKS, WEEK,
        _fold_qcp_matrices, score_by_issue_day, score_predictions,
    )
    from .outside_hub_candidate_models import make_candidate
    from .qcp_heating_feature import hourly_from_baseline
except ImportError:  # direct script invocation from repository root
    from evaluate_outside_hub import (
        CV_DAY_OFFSETS, DAY, SIMULATION_START, supervised_rows,
    )
    from evaluate_outside_hub_with_er import (
        CORRECTED_BASELINE_DIR, DEFAULT_OUTPUT as CORE_DIR, _sha256,
        load_composite_target,
    )
    from evaluate_simulated_home_hub import (
        HORIZON_STEPS, INTERVAL, QCP_COLUMNS, TRAIN_WEEKS, WEEK,
        _fold_qcp_matrices, score_by_issue_day, score_predictions,
    )
    from outside_hub_candidate_models import make_candidate
    from qcp_heating_feature import hourly_from_baseline


DEFAULT_OUTPUT = CORE_DIR / "supplemental/rf_history_hub_qcp"
METHODS = (
    "rf_history_weather_lag", "rf_history_hub_qcp",
    "rf_history_weather_hub_qcp",
)
LEAVES = (5, 15)
WEATHER_COLUMNS = ("ambient_lag_1_c", "ambient_lag_1_available")
KEY = ["train_weeks", "origin", "target", "lead"]


def _read_hub_and_weather(corrected_baseline_dir: Path,
                          target_index: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict[str, str]]:
    meter_path = corrected_baseline_dir / "data/home_hub_meter_15min.csv"
    weather_path = corrected_baseline_dir / "data/realized_weather_15min.csv"
    meter = pd.read_csv(meter_path, parse_dates=["timestamp"]).set_index("timestamp")
    weather = pd.read_csv(weather_path, parse_dates=["timestamp"]).set_index("timestamp")
    if (not meter.index.equals(target_index) or not weather.index.equals(target_index)
            or meter.index.has_duplicates or weather.index.has_duplicates):
        raise ValueError("Corrected Hub meter and weather must match the target's six-week grid")
    data = meter[["home_hub_power_kw", "ev_power_kw"]].join(
        weather[["ambient_temperature_c"]], how="inner", validate="one_to_one"
    )
    if (not data.index.equals(target_index)
            or not np.isfinite(data.to_numpy(float)).all()):
        raise ValueError("Hub QCP inputs must be finite on the full target grid")
    return data, {"hub_meter_15min_sha256": _sha256(meter_path),
                  "realized_weather_15min_sha256": _sha256(weather_path)}


def _weather_matrix(rows, weather: pd.Series) -> pd.DataFrame:
    """Append only observed ambient at t-15 minutes to the old history set."""
    prior = weather.reindex(rows.origin - INTERVAL).to_numpy(float)
    available = np.isfinite(prior)
    x = rows.x.loc[:, list(rows.feature_sets["history"])].copy()
    x[WEATHER_COLUMNS[0]] = np.where(available, prior, 0.0)
    x[WEATHER_COLUMNS[1]] = available.astype(int)
    if not np.isfinite(x.to_numpy(float)).all():
        raise ValueError("Weather ablation contains nonfinite features")
    return x


def _model_matrices(method: str, fit_mask: np.ndarray, valid_mask: np.ndarray,
                    qtrain: pd.DataFrame, qvalid: pd.DataFrame,
                    weather_x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if method == "rf_history_weather_lag":
        return weather_x.loc[fit_mask], weather_x.loc[valid_mask]
    if method == "rf_history_hub_qcp":
        return qtrain, qvalid
    if method == "rf_history_weather_hub_qcp":
        return (
            qtrain.join(weather_x.loc[fit_mask, list(WEATHER_COLUMNS)], validate="one_to_one"),
            qvalid.join(weather_x.loc[valid_mask, list(WEATHER_COLUMNS)], validate="one_to_one"),
        )
    raise ValueError(f"Unknown supplemental RF method: {method}")


def _candidate_forecast(model: Any, x: pd.DataFrame) -> np.ndarray:
    result = np.asarray(model.predict(x), dtype=float)
    if not np.isfinite(result).all():
        raise ValueError("RF forecast contains nonfinite values")
    return np.maximum(result, 0.0)


def _forecast_rows(rows, mask: np.ndarray, values: np.ndarray,
                   method: str, weeks: int) -> pd.DataFrame:
    if len(values) != int(mask.sum()):
        raise ValueError("Forecast count differs from selected origin-target pairs")
    return pd.DataFrame({
        "method": method, "train_weeks": weeks, "split": "test",
        "origin": rows.origin[mask], "target": rows.target[mask],
        "lead": rows.lead[mask], "forecast_kw": values,
        "actual_kw": rows.y[mask],
    })


def _saved_rf(core_dir: Path, rows, test_mask: np.ndarray,
              target_audit: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Require saved plain RF to use exactly this truth, mask, and test grid."""
    metadata_path = core_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (metadata.get("target") != target_audit["target"]
            or metadata.get("observed_bins") != target_audit["observed_bins"]
            or metadata.get("masked_bins") != target_audit["masked_bins"]
            or metadata.get("raw_ochre_sha256") != target_audit["raw_ochre_sha256"]
            or metadata.get("corrected_hub_meter_sha256")
            != target_audit["corrected_hub_meter_sha256"]
            or metadata.get("training_weeks") != list(TRAIN_WEEKS)
            or metadata.get("cv_fold_day_offsets_in_week5") != list(CV_DAY_OFFSETS)):
        raise ValueError("Saved plain RF benchmark is from a different study")
    path = core_dir / "predictions.csv.gz"
    base = pd.read_csv(path, parse_dates=["origin", "target"])
    base = base.loc[base["method"].eq("rf_history") & base["split"].eq("test")]
    base = base.sort_values(KEY).reset_index(drop=True)
    expected = pd.concat([
        _forecast_rows(rows, test_mask, np.full(int(test_mask.sum()), np.nan),
                       "rf_history", weeks)
        for weeks in TRAIN_WEEKS
    ], ignore_index=True).sort_values(KEY).reset_index(drop=True)
    if (len(base) != len(expected) or not base[KEY].equals(expected[KEY])
            or not np.allclose(base["actual_kw"], expected["actual_kw"], rtol=0, atol=1e-8)
            or not np.isfinite(base["forecast_kw"]).all()):
        raise ValueError("Saved plain RF forecasts do not match this exact test grid/truth")
    return base, {"saved_rf_predictions_sha256": _sha256(path),
                  "saved_rf_metadata_sha256": _sha256(metadata_path),
                  "saved_rf_selected_params": {
                      f"{weeks}w": metadata["selected_validation_params"][f"{weeks}w:rf_history"]
                      for weeks in TRAIN_WEEKS
                  }}


def _common_origins(core_dir: Path) -> tuple[pd.DatetimeIndex, dict[str, Any]]:
    """Use the same 640 causal issues as the saved hybrid comparison."""
    path = core_dir / "hybrid_predictions.csv.gz"
    hybrid = pd.read_csv(path, usecols=["split", "origin", "lead"],
                         parse_dates=["origin"])
    hybrid = hybrid.loc[hybrid["split"].eq("test")]
    groups = hybrid.groupby("origin")["lead"].agg(["size", "min", "max"])
    if (len(groups) != 640 or not groups["size"].eq(HORIZON_STEPS).all()
            or not groups["min"].eq(1).all()
            or not groups["max"].eq(HORIZON_STEPS).all()):
        raise ValueError("Saved hybrid reference no longer has 640 complete test issues")
    return pd.DatetimeIndex(groups.index), {
        "saved_hybrid_predictions_sha256": _sha256(path),
        "common_test_origins": len(groups),
    }


def _coverage_row(weeks: int, phase: str, start: pd.Timestamp,
                  cutoff: pd.Timestamp, info: dict[str, Any],
                  train_x: pd.DataFrame, valid_x: pd.DataFrame) -> dict[str, Any]:
    fit = info["evaluation_fit"]
    training = info["training_feature_info"]
    return {
        "train_weeks": weeks, "phase": phase,
        "fit_start": start, "fit_cutoff_exclusive": cutoff,
        "qcp_fit_available": bool(fit["fit_available"]),
        "qcp_eligible_hour_groups": int(fit["eligible_hour_groups"]),
        "qcp_training_hours": int(fit["training_hours"]),
        "train_feature_available_pairs": int(train_x[QCP_COLUMNS[1]].sum()),
        "train_feature_total_pairs": len(train_x),
        "train_feature_available_origins": int(training["available_origins"]),
        "train_feature_total_origins": int(training["total_origins"]),
        "evaluation_feature_available_pairs": int(valid_x[QCP_COLUMNS[1]].sum()),
        "evaluation_feature_total_pairs": len(valid_x),
        "evaluation_feature_available_origins": int(
            valid_x[QCP_COLUMNS[1]].sum() // HORIZON_STEPS
        ),
        "evaluation_feature_total_origins": len(valid_x) // HORIZON_STEPS,
        "unavailable_training_days": json.dumps(training["unavailable_days"], sort_keys=True),
    }


def run(*, core_dir: Path = CORE_DIR,
        corrected_baseline_dir: Path = CORRECTED_BASELINE_DIR,
        output_dir: Path = DEFAULT_OUTPUT) -> None:
    core_dir = core_dir.expanduser().resolve()
    corrected_baseline_dir = corrected_baseline_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    truth, observed, target_audit = load_composite_target(
        corrected_baseline_dir=corrected_baseline_dir,
    )
    rows, observed_labels = supervised_rows(observed, truth)
    data, input_hashes = _read_hub_and_weather(corrected_baseline_dir, truth.index)
    hourly = hourly_from_baseline(data)
    weather_x = _weather_matrix(rows, data["ambient_temperature_c"])
    test_start, test_end = SIMULATION_START + 5 * WEEK, SIMULATION_START + 6 * WEEK
    validation_start = SIMULATION_START + 4 * WEEK
    test_mask = rows.mask_evaluation(test_start, test_end)
    observed_target = np.isfinite(observed_labels)
    if int(test_mask.sum()) != 641 * HORIZON_STEPS:
        raise ValueError("Unexpected direct-model held-out test issue count")
    baseline, base_provenance = _saved_rf(core_dir, rows, test_mask, target_audit)
    common_origins, hybrid_provenance = _common_origins(core_dir)
    if not common_origins.isin(pd.DatetimeIndex(rows.origin[test_mask].unique())).all():
        raise ValueError("Hybrid reference issue grid differs from direct-model test grid")

    cv_records: list[dict[str, Any]] = []
    coverage_records: list[dict[str, Any]] = []
    selected: dict[str, dict[str, Any]] = {}
    forecasts = [baseline]
    for weeks in TRAIN_WEEKS:
        fold_cache: dict[int, dict[str, Any]] = {}
        for offset in CV_DAY_OFFSETS:
            fold_start = validation_start + offset * DAY
            fold_end = fold_start + DAY
            train_start = fold_start - weeks * WEEK
            fit = rows.mask_fit(train_start, fold_start) & observed_target
            valid = rows.mask_evaluation(fold_start, fold_end)
            if not fit.any() or not valid.any():
                raise ValueError(f"Empty {weeks}-week fold at {fold_start}")
            qtrain, qvalid, info = _fold_qcp_matrices(
                hourly, rows, fit, valid, train_start, fold_start
            )
            coverage_records.append(_coverage_row(
                weeks, f"validation_day_{offset}", train_start, fold_start,
                info, qtrain, qvalid,
            ))
            fold_cache[offset] = {
                "fit": fit, "valid": valid, "qtrain": qtrain,
                "qvalid": qvalid, "fold_start": fold_start,
                "fold_end": fold_end, "train_start": train_start,
            }
            print(f"Prepared {weeks}w QCP fold {offset}", flush=True)

        final_start = test_start - weeks * WEEK
        final_fit = rows.mask_fit(final_start, test_start) & observed_target
        qtrain_final, qtest, final_info = _fold_qcp_matrices(
            hourly, rows, final_fit, test_mask, final_start, test_start
        )
        coverage_records.append(_coverage_row(
            weeks, "test_fit", final_start, test_start,
            final_info, qtrain_final, qtest,
        ))
        print(f"Prepared {weeks}w final QCP fit", flush=True)

        for method in METHODS:
            cv_maes: list[float] = []
            for candidate, leaf in enumerate(LEAVES, 1):
                daily_maes: list[float] = []
                for offset in CV_DAY_OFFSETS:
                    part = fold_cache[offset]
                    fit, valid = part["fit"], part["valid"]
                    xfit, xvalid = _model_matrices(
                        method, fit, valid, part["qtrain"], part["qvalid"], weather_x
                    )
                    model = make_candidate("rf_history", {"min_samples_leaf": leaf})
                    model.fit(xfit, observed_labels[fit])
                    pred = _candidate_forecast(model, xvalid)
                    mae = float(np.mean(np.abs(pred - rows.y[valid])))
                    daily_maes.append(mae)
                    cv_records.append({
                        "train_weeks": weeks, "method": method,
                        "candidate": candidate, "fold_start": part["fold_start"],
                        "fold_end": part["fold_end"], "fit_start": part["train_start"],
                        "fit_end_exclusive": part["fold_start"],
                        "fit_pairs": int(fit.sum()), "validation_pairs": int(valid.sum()),
                        "mae_kw": mae,
                        "params": json.dumps({"min_samples_leaf": leaf}, sort_keys=True),
                    })
                cv_maes.append(float(np.mean(daily_maes)))
                print(f"{weeks}w {method} leaf={leaf}: CV {cv_maes[-1]:.4f}", flush=True)
            best = int(np.argmin(cv_maes))
            leaf = LEAVES[best]
            selected[f"{weeks}w:{method}"] = {
                "params": {"min_samples_leaf": leaf},
                "cv_mae_kw": cv_maes[best], "cv_grid_mae_kw": cv_maes,
            }
            xfit, xtest = _model_matrices(
                method, final_fit, test_mask, qtrain_final, qtest, weather_x
            )
            model = make_candidate("rf_history", {"min_samples_leaf": leaf})
            model.fit(xfit, observed_labels[final_fit])
            pred = _candidate_forecast(model, xtest)
            forecasts.append(_forecast_rows(rows, test_mask, pred, method, weeks))
            print(f"{weeks}w {method}: test issued", flush=True)

    all_predictions = pd.concat(forecasts, ignore_index=True)
    all_predictions = all_predictions.sort_values(["train_weeks", "method", "origin", "lead"])
    matched = all_predictions.loc[all_predictions["origin"].isin(common_origins)]
    for name, frame, origins in (("full", all_predictions, 641),
                                 ("matched", matched, 640)):
        counts = frame.groupby(["train_weeks", "method"])["origin"].nunique()
        if len(counts) != 3 * (len(METHODS) + 1) or not counts.eq(origins).all():
            raise ValueError(f"{name} test predictions have unequal issue coverage")
    threshold_kw = float(json.loads((core_dir / "metadata.json").read_text())[
        "peak_threshold_kw"
    ])
    full_scores, _ = score_predictions(all_predictions, threshold_kw)
    matched_scores, lead_scores = score_predictions(matched, threshold_kw)
    output_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_dir / "predictions.csv.gz", "wt", encoding="utf-8") as stream:
        all_predictions.to_csv(stream, index=False)
    pd.DataFrame(cv_records).to_csv(output_dir / "rolling_cv.csv", index=False)
    pd.DataFrame(coverage_records).to_csv(output_dir / "qcp_fold_coverage.csv", index=False)
    full_scores.to_csv(output_dir / "full_horizon_scores.csv", index=False)
    matched_scores.to_csv(output_dir / "matched_horizon_scores.csv", index=False)
    lead_scores.to_csv(output_dir / "matched_lead_scores.csv", index=False)
    score_by_issue_day(matched, threshold_kw).to_csv(
        output_dir / "matched_issue_day_scores.csv", index=False,
    )
    metadata = {
        "target": target_audit["target"],
        "target_observation_mask": "originally observed Outside-Hub source bins only",
        "qcp_source": "corrected Home Hub minus separately metered EV; latest completed hour",
        "qcp_role": "historical cross-meter HVAC heating proxy, not direct ER disaggregation",
        "qcp_formulation": "existing approved 24-hour-group symmetric heating QCP, fit within fold",
        "weather_ablation": "same RF history features plus observed ambient at t-minus-15-minutes",
        "combined_ablation": "RF history plus both past ambient and approved Hub QCP heating proxy",
        "rf_spec": "80 trees, 0.8 max_features, 0.8 max_samples, bootstrap, seed 42",
        "rf_leaf_grid": LEAVES,
        "training_weeks": TRAIN_WEEKS,
        "cv_fold_day_offsets_in_week5": CV_DAY_OFFSETS,
        "test_interval": [test_start.isoformat(), test_end.isoformat()],
        "full_test_origins": 641, "matched_test_origins": 640,
        "peak_threshold_kw": threshold_kw,
        "selected_validation_params": selected,
        **input_hashes, **base_provenance, **hybrid_provenance,
        "source_module_sha256": _sha256(Path(__file__)),
        "qcp_module_sha256": _sha256(Path(__file__).with_name("qcp_heating_feature.py")),
        "package_versions": {name: importlib.metadata.version(name)
                             for name in ("numpy", "pandas", "scikit-learn")},
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8"
    )
    _write_report(output_dir, matched_scores, selected,
                  pd.DataFrame(coverage_records),
                  base_provenance["saved_rf_selected_params"])
    print(f"Saved supplemental RF comparison to {output_dir}", flush=True)


def _write_report(output_dir: Path, scores: pd.DataFrame,
                  selected: dict[str, dict[str, Any]],
                  coverage: pd.DataFrame,
                  base_selected: dict[str, dict[str, Any]]) -> None:
    eight = scores.loc[scores["scope"].eq("through_horizon")
                       & scores["horizon_hours"].eq(8)].set_index(
        ["train_weeks", "method"]
    )
    lines = [
        "# EV-only Outside-Hub plus resistance: RF history and QCP follow-up", "",
        "The earlier comparison **already included RF with load history**. This "
        "supplement compares that saved model with (a) the same RF given the latest "
        "completed observed ambient reading, (b) the same RF given the existing "
        "approved Home Hub QCP heating feature, and (c) both features together. "
        "The combined model measures QCP's incremental effect beyond raw past "
        "temperature. The QCP source is the corrected "
        "Home Hub meter minus the EV meter, and is a cross-meter heating proxy, "
        "not direct electric-resistance disaggregation.", "",
        "All methods use originally observed Outside-Hub-plus-resistance target "
        "labels for training, trailing 2/3/4-week windows, four chronological "
        "week-five validation day folds, and held-out week six. Parameters are "
        "selected by mean validation MAE. The saved RF baseline is verified against "
        "the exact target, test grid, and study provenance. Headline scores cover "
        "the same 640 eight-hour test issues as the existing hybrid comparison.", "",
        "| Training | RF features | Week-5 CV MAE (kW) | Week-6 8h MAE (kW) | "
        "Bias (kW) | Underforecast | Peak recall | Peak shortfall (kW) |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for weeks in TRAIN_WEEKS:
        for method in ("rf_history", *METHODS):
            row = eight.loc[(weeks, method)]
            choice = (base_selected[f"{weeks}w"] if method == "rf_history"
                      else selected[f"{weeks}w:{method}"])
            lines.append(
                f"| {weeks} weeks | {method} | {choice['cv_mae_kw']:.3f} | "
                f"{row['mae_kw']:.3f} | {row['bias_kw']:+.3f} | "
                f"{row['underforecast_rate']:.3f} | {row['peak_recall']:.3f} | "
                f"{row['actual_peak_mean_shortfall_kw']:.3f} |"
            )
    final = coverage.loc[coverage["phase"].eq("test_fit")]
    plain_4w = float(eight.loc[(4, "rf_history"), "mae_kw"])
    weather_4w = float(eight.loc[(4, "rf_history_weather_lag"), "mae_kw"])
    qcp_4w = float(eight.loc[(4, "rf_history_hub_qcp"), "mae_kw"])
    combined_4w = float(eight.loc[(4, "rf_history_weather_hub_qcp"), "mae_kw"])
    lines += ["", f"At four weeks, past ambient reduced eight-hour MAE from "
              f"{plain_4w:.3f} to {weather_4w:.3f} kW. Hub QCP alone scored "
              f"{qcp_4w:.3f} kW; adding it on top of past ambient scored "
              f"{combined_4w:.3f} kW. The QCP proxy therefore did not improve "
              "point accuracy beyond past ambient on this held-out week.", "",
              "Every final Hub QCP fit had "
              f"{int(final['qcp_eligible_hour_groups'].min())}/24 eligible local "
              "hour groups. Early walk-forward training days without a fit remain "
              "in the RF sample with an unavailable flag and zero-filled QCP value.", "",
              "Peak means a 15-minute target mean at or above the fixed p95 of "
              "originally observed adjusted-target bins in weeks 1–4. There are "
              "only seven distinct week-six peak bins; overlapping forecasts "
              "repeat those bins across issue-target pairs. Peak metrics are "
              "therefore descriptive, with limited event count.", "",
              "[Matched scores](matched_horizon_scores.csv), "
              "[lead scores](matched_lead_scores.csv), "
              "[all 641 direct issues](full_horizon_scores.csv), "
              "[forecasts](predictions.csv.gz), "
              "[validation folds](rolling_cv.csv), "
              "[QCP coverage](qcp_fold_coverage.csv), and "
              "[provenance](metadata.json).", ""]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-dir", type=Path, default=CORE_DIR)
    parser.add_argument("--corrected-baseline-dir", type=Path,
                        default=CORRECTED_BASELINE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
