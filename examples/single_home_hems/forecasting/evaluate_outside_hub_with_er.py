"""Compare EV-only Outside-Hub forecasts with HVAC resistance outside the Hub.

The original six-week OCHRE export placed HVAC electric resistance inside its
Home Hub meter. The corrected EV-only meter projection removes resistance from
the Hub without changing the simulated plant. This parallel study forecasts
fixed Outside-Hub load plus realized HVAC electric resistance, equivalently
total-home minus the corrected Home Hub meter.

Candidate features and fit labels use the composite target only where the
fixed Outside-Hub source was originally observed. The contemporaneous OCHRE
resistance contribution is combined into that aggregate measurement; it is
never supplied as a separate feature. The complete replay is scoring truth.
This study preserves the existing direct-model features, four chronological
week-five CV folds, two/three/four-week fit windows, and held-out week six.
Peak and hybrid-reference formulations are intentionally outside this module.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .evaluate_outside_hub import (
        BASELINE_DIR, DEFAULT_OUTPUT as FIXED_STUDY_DIR,
        EXPECTED_BLOCKS, EXPECTED_MINUTES, MINUTE, SIMULATION_START,
        load_replayed_truth, observed_source_on_simulation_clock,
        selected_families, supervised_rows, train_candidates,
    )
    from .evaluate_simulated_home_hub import (
        HORIZON_STEPS, TRAIN_WEEKS, WEEK, score_by_issue_day, score_predictions,
    )
except ImportError:  # direct script invocation from the repository root
    from evaluate_outside_hub import (
        BASELINE_DIR, DEFAULT_OUTPUT as FIXED_STUDY_DIR,
        EXPECTED_BLOCKS, EXPECTED_MINUTES, MINUTE, SIMULATION_START,
        load_replayed_truth, observed_source_on_simulation_clock,
        selected_families, supervised_rows, train_candidates,
    )
    from evaluate_simulated_home_hub import (
        HORIZON_STEPS, TRAIN_WEEKS, WEEK, score_by_issue_day, score_predictions,
    )


CORRECTED_BASELINE_DIR = BASELINE_DIR.parent / "ev_home_hub_baseline_6wk_no_er"
DEFAULT_OUTPUT = CORRECTED_BASELINE_DIR / "outside_hub_with_resistance_forecast_study"
ER_COLUMN = "HVAC Heating ER Power (kW)"
TARGET_NAME = "outside_hub_with_hvac_er_kw"
TOLERANCE_KW = 1e-7


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _meter(path: Path, column: str) -> pd.Series:
    frame = pd.read_csv(path, usecols=["timestamp", column], parse_dates=["timestamp"])
    index = pd.DatetimeIndex(frame["timestamp"], name="timestamp")
    values = pd.to_numeric(frame[column], errors="raise").to_numpy(float)
    return pd.Series(values, index=index, name=column)


def _require_same_grid(series: pd.Series, reference: pd.DatetimeIndex,
                       label: str) -> None:
    if (len(series) != EXPECTED_MINUTES or not series.index.equals(reference)
            or not np.isfinite(series.to_numpy(float)).all()):
        raise ValueError(f"{label} must be finite on the exact six-week minute grid")


def load_composite_target(
    *, baseline_dir: Path = BASELINE_DIR,
    corrected_baseline_dir: Path = CORRECTED_BASELINE_DIR,
    fixed_study_dir: Path = FIXED_STUDY_DIR,
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    """Return complete corrected-wiring truth and masked aggregate observations."""
    baseline_dir = baseline_dir.expanduser().resolve()
    corrected_baseline_dir = corrected_baseline_dir.expanduser().resolve()
    fixed_study_dir = fixed_study_dir.expanduser().resolve()
    fixed_truth, fixed_audit = load_replayed_truth(baseline_dir)

    prior_metadata_path = fixed_study_dir / "metadata.json"
    prior_metadata = json.loads(prior_metadata_path.read_text(encoding="utf-8"))
    if prior_metadata.get("fast"):
        raise ValueError("Composite study needs the full original Outside-Hub study")
    source_path = Path(prior_metadata["source_csv"])
    if not source_path.is_file():
        source_path = fixed_study_dir / "outside_hub_forecast_source.csv"
    if _sha256(source_path) != prior_metadata["source_sha256"]:
        raise ValueError("Original Outside-Hub source differs from the saved study")
    fixed_observed, source_audit = observed_source_on_simulation_clock(
        source_path, fixed_truth,
    )
    prior_15min = pd.read_csv(
        fixed_study_dir / "outside_hub_15min.csv", parse_dates=["timestamp"],
    ).set_index("timestamp")
    if (not prior_15min.index.equals(fixed_truth.index)
            or not np.allclose(prior_15min["outside_hub_replayed_kw"],
                               fixed_truth, rtol=0, atol=1e-10)
            or not np.allclose(
                prior_15min["outside_hub_originally_observed_kw"],
                fixed_observed, rtol=0, atol=1e-10, equal_nan=True,
            )):
        raise ValueError("Original Outside-Hub replay or observation mask changed")

    raw_path = baseline_dir / "data/raw/home_01.csv"
    old_hub_path = baseline_dir / "data/home_hub_meter_minute.csv"
    corrected_hub_path = corrected_baseline_dir / "data/home_hub_meter_minute.csv"
    whole_path = baseline_dir / "data/home_01_standardized.csv"
    corrected_metadata = json.loads(
        (corrected_baseline_dir / "baseline_metadata.json").read_text(encoding="utf-8")
    )
    if (corrected_metadata.get("source_raw_sha256") != _sha256(raw_path)
            or corrected_metadata.get("source_meter_sha256") != _sha256(old_hub_path)):
        raise ValueError("Corrected EV-only Hub meter has different source provenance")

    old_hub = _meter(old_hub_path, "home_hub_power_kw")
    corrected_hub = _meter(corrected_hub_path, "home_hub_power_kw")
    total = _meter(whole_path, "total_electric_power_kw")
    fixed_outside = _meter(whole_path, "outside_hub_power_kw")
    index = pd.date_range(SIMULATION_START, periods=EXPECTED_MINUTES,
                          freq=MINUTE, name="timestamp")
    for label, series in (("Original Hub", old_hub),
                          ("Corrected Hub", corrected_hub),
                          ("Total-home", total),
                          ("Fixed Outside-Hub", fixed_outside)):
        _require_same_grid(series, index, label)

    raw = pd.read_csv(raw_path, usecols=["Time", ER_COLUMN], parse_dates=["Time"])
    raw_index = pd.DatetimeIndex(raw["Time"], name="timestamp")
    if (raw_index.hasnans or raw_index.has_duplicates
            or not raw_index.is_monotonic_increasing):
        raise ValueError("Raw OCHRE resistance timestamps are invalid")
    er = pd.Series(pd.to_numeric(raw[ER_COLUMN], errors="raise").to_numpy(float),
                   index=raw_index, name="hvac_er_kw").reindex(index)
    _require_same_grid(er, index, "HVAC electric resistance")
    if er.lt(-TOLERANCE_KW).any():
        raise ValueError("HVAC electric resistance cannot be negative")
    er = er.clip(lower=0.0)

    hub_error = float(np.max(np.abs(old_hub - corrected_hub - er)))
    balance_error = float(np.max(np.abs(total - corrected_hub - fixed_outside - er)))
    fixed_error = float(np.max(np.abs(total - old_hub - fixed_outside)))
    if max(hub_error, balance_error, fixed_error) > TOLERANCE_KW:
        raise ValueError("Corrected EV-only wiring fails total/Hub/ER power balance")
    composite_minute = (fixed_outside + er).rename(TARGET_NAME)
    grouped = composite_minute.resample("15min", label="left", closed="left")
    truth = grouped.mean()
    er_15min = er.resample("15min", label="left", closed="left").mean()
    if (len(truth) != EXPECTED_BLOCKS or not truth.index.equals(fixed_truth.index)
            or not grouped.count().eq(15).all()
            or not np.allclose(truth, fixed_truth + er_15min, rtol=0, atol=1e-10)
            or not np.isfinite(truth.to_numpy(float)).all()):
        raise ValueError("Composite 15-minute truth is incomplete or imbalanced")
    observed = (fixed_observed + er_15min).where(fixed_observed.notna()).rename(
        f"{TARGET_NAME}_originally_observed"
    )
    known = observed.notna()
    if (not known.equals(fixed_observed.notna())
            or not np.allclose(observed.loc[known], truth.loc[known],
                               rtol=0, atol=1e-6)):
        raise ValueError("Composite observations disagree with replay or source mask")
    audit = {
        "target": "outside_hub_with_resistance_ev_only",
        "target_description": "EV-only Outside-Hub forecast with HVAC electric resistance",
        "truth_equation": "fixed Outside-Hub + OCHRE HVAC Heating ER = total-home - corrected Home Hub",
        "observed_equation": (
            "originally observed fixed Outside-Hub + contemporaneous OCHRE HVAC ER; "
            "retain original Outside-Hub availability mask"
        ),
        "features": "only lagged aggregate target observations and calendar/lead features",
        "raw_ochre_sha256": _sha256(raw_path),
        "original_hub_meter_sha256": _sha256(old_hub_path),
        "corrected_hub_meter_sha256": _sha256(corrected_hub_path),
        "fixed_outside_source_sha256": _sha256(source_path),
        "fixed_outside_study_metadata_sha256": _sha256(prior_metadata_path),
        "observed_bins": int(known.sum()),
        "masked_bins": int((~known).sum()),
        "resistance_positive_minutes": int(er.gt(0).sum()),
        "resistance_mean_kw": float(er.mean()),
        "resistance_max_kw": float(er.max()),
        "max_original_minus_corrected_hub_minus_er_kw": hub_error,
        "max_total_minus_corrected_hub_minus_target_kw": balance_error,
        "max_total_minus_original_hub_minus_fixed_outside_kw": fixed_error,
        "fixed_outside_truth_audit": fixed_audit,
        "fixed_outside_source_audit": source_audit,
    }
    return truth, observed, audit


def write_report(output_dir: Path, *, cv: pd.DataFrame,
                 scores: pd.DataFrame, selected: dict[str, dict[str, Any]],
                 audit: dict[str, Any], threshold_kw: float,
                 test_peak_bins: int) -> None:
    choices = selected_families(selected)
    eight = scores.loc[
        scores["horizon_hours"].eq(8) & scores["scope"].eq("through_horizon")
    ].set_index(
        ["train_weeks", "method"]
    )
    lines = [
        "# EV-only Outside-Hub forecast with HVAC electric resistance", "",
        "Target: fixed Outside-Hub replay load plus live OCHRE HVAC electric "
        "resistance. This equals total-home power minus the corrected EV-only "
        "Home Hub meter. The original Outside-Hub study excluded resistance "
        "because its saved Hub meter included that circuit.", "",
        "Only the aggregate composite is an input to these direct predictors. "
        "The resistance channel is used offline to construct and audit the "
        "corrected meter; it is never passed as a separate feature. Source "
        "gaps remain masked. Complete OCHRE replay values are scoring truth.", "",
        "Two-, three-, and four-week trailing fits use the same chronological "
        "week-five folds and held-out week-six issues as the original "
        "Outside-Hub study. Parameter selection uses only week-five CV MAE. "
        "The models are refitted immediately before week six. The RF-to-p90 "
        "hybrid is evaluated separately.", "",
        f"An actual peak is a 15-minute aggregate mean at least {threshold_kw:.6f} kW: "
        "the 95th percentile of originally observed adjusted-target bins in "
        "weeks 1–4, fixed before validation. The week-six replay contains "
        f"{test_peak_bins} distinct peak bins. Recall and shortfall use overlapping "
        "issue-target pairs, so one peak bin can be counted many times.", "",
        f"Observed aggregate bins: {audit['observed_bins']}/{EXPECTED_BLOCKS}; "
        f"HVAC resistance-positive minutes: {audit['resistance_positive_minutes']}/{EXPECTED_MINUTES}.", "",
        "| Model | Training | Week-5 CV MAE (kW) | Week-6 8h MAE (kW) | "
        "Bias (kW) | Underforecast | Peak recall | Peak shortfall (kW) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for weeks in TRAIN_WEEKS:
        for method in sorted(scores["method"].unique()):
            result = selected[f"{weeks}w:{method}"]
            row = eight.loc[(weeks, method)]
            lines.append(
                f"| {method} | {weeks} weeks | {result['cv_mae_kw']:.4f} | "
                f"{row['mae_kw']:.4f} | {row['bias_kw']:+.4f} | "
                f"{row['underforecast_rate']:.3f} | {row['peak_recall']:.3f} | "
                f"{row['actual_peak_mean_shortfall_kw']:.4f} |"
            )
    lines += [
        "", "Week-five selected family by training window: "
        + ", ".join(f"{weeks}w {choices[weeks]}" for weeks in TRAIN_WEEKS) + ".", "",
        "Scores count overlapping issued eight-hour forecasts; shared target "
        "intervals are therefore repeated across issues. The EV controller "
        "is unchanged. Files: [predictions](predictions.csv.gz), "
        "[validation folds](rolling_cv.csv), "
        "[horizon scores](horizon_scores.csv), "
        "[lead scores](lead_scores.csv), "
        "[issue-day scores](issue_day_scores.csv), "
        "[target and mask](outside_hub_15min.csv), "
        "[provenance](metadata.json).", "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run(*, baseline_dir: Path = BASELINE_DIR,
        corrected_baseline_dir: Path = CORRECTED_BASELINE_DIR,
        fixed_study_dir: Path = FIXED_STUDY_DIR,
        output_dir: Path = DEFAULT_OUTPUT, fast: bool = False) -> None:
    truth, observed, audit = load_composite_target(
        baseline_dir=baseline_dir,
        corrected_baseline_dir=corrected_baseline_dir,
        fixed_study_dir=fixed_study_dir,
    )
    threshold_history = observed.loc[
        (observed.index >= SIMULATION_START)
        & (observed.index < SIMULATION_START + 4 * WEEK)
    ].dropna()
    if threshold_history.empty:
        raise ValueError("Peak threshold needs originally observed weeks 1–4")
    threshold_kw = float(threshold_history.quantile(0.95))
    test_truth = truth.loc[
        (truth.index >= SIMULATION_START + 5 * WEEK)
        & (truth.index < SIMULATION_START + 6 * WEEK)
    ]
    test_peak_bins = int(test_truth.ge(threshold_kw).sum())
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / "outside_hub_15min.csv"
    pd.DataFrame({
        "timestamp": truth.index,
        "outside_hub_replayed_kw": truth.to_numpy(float),
        "outside_hub_originally_observed_kw": observed.to_numpy(float),
        "source_observed": observed.notna().to_numpy(),
    }).to_csv(target_path, index=False)
    (output_dir / "target_metadata.json").write_text(json.dumps({
        "target": audit["target"],
        "target_description": audit["target_description"],
        "peak_threshold_kw": threshold_kw,
        "peak_threshold_rule": "p95 of originally observed adjusted-target bins in weeks 1–4",
        "matched_test_origins": 7 * 96 - HORIZON_STEPS + 1,
        "source_15min_sha256": _sha256(target_path),
        "target_audit": audit,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared corrected Outside-Hub+ER source at {target_path}", flush=True)
    rows, observed_labels = supervised_rows(observed, truth)
    cv, predictions, selected = train_candidates(rows, observed_labels, fast=fast)
    if not fast:
        expected_origins = 7 * 96 - HORIZON_STEPS + 1
        for (weeks, method), part in predictions.groupby(["train_weeks", "method"]):
            if (part["origin"].nunique() != expected_origins
                    or len(part) != expected_origins * HORIZON_STEPS
                    or not part["split"].eq("test").all()):
                raise ValueError(f"{weeks}w {method} lacks full matched test issues")
    scores, lead_scores = score_predictions(predictions, threshold_kw)
    with gzip.open(output_dir / "predictions.csv.gz", "wt", encoding="utf-8") as stream:
        predictions.to_csv(stream, index=False)
    cv.to_csv(output_dir / "rolling_cv.csv", index=False)
    scores.to_csv(output_dir / "horizon_scores.csv", index=False)
    lead_scores.to_csv(output_dir / "lead_scores.csv", index=False)
    score_by_issue_day(predictions, threshold_kw).to_csv(
        output_dir / "issue_day_scores.csv", index=False,
    )
    metadata = {
        **audit,
        "simulation_start": SIMULATION_START.isoformat(),
        "validation_start": (SIMULATION_START + 4 * WEEK).isoformat(),
        "test_start": (SIMULATION_START + 5 * WEEK).isoformat(),
        "test_end": (SIMULATION_START + 6 * WEEK).isoformat(),
        "interval_minutes": 15,
        "horizon_steps": HORIZON_STEPS,
        "training_weeks": TRAIN_WEEKS,
        "cv_fold_day_offsets_in_week5": [0] if fast else [0, 2, 4, 6],
        "validation_selection_metric": "mean four-fold eight-hour MAE",
        "test_evaluation": "held-out week 6, full eight-hour issues within week",
        "test_origins_per_candidate": int(predictions["origin"].nunique()),
        "matched_test_origins": int(predictions["origin"].nunique()),
        "source_15min_sha256": _sha256(target_path),
        "peak_threshold_kw": threshold_kw,
        "peak_threshold_rule": "p95 of originally observed adjusted-target bins in weeks 1–4",
        "test_distinct_peak_bins": test_peak_bins,
        "selected_validation_params": selected,
        "selected_validation_methods": selected_families(selected),
        "hybrid_reference": "evaluated in a separate RF28d-to-p90_21d study",
        "package_versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scikit-learn", "lightgbm", "xgboost", "catboost")
        },
        "fast": fast,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    write_report(output_dir, cv=cv, scores=scores, selected=selected, audit=audit,
                 threshold_kw=threshold_kw, test_peak_bins=test_peak_bins)
    print(f"Saved EV-only Outside-Hub+ER direct-model study to {output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=BASELINE_DIR)
    parser.add_argument("--corrected-baseline-dir", type=Path,
                        default=CORRECTED_BASELINE_DIR)
    parser.add_argument("--fixed-study-dir", type=Path, default=FIXED_STUDY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fast", action="store_true",
                        help="One CV fold/setting and hourly test origins")
    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
