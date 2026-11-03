"""Compare causal Outside-Hub forecasts on the six-week OCHRE baseline.

The target is 15-minute mean Total-home minus Home-Hub electricity. Candidate
features use only complete, originally observed Outside-Hub source bins before
each issue; missing historical bins have availability flags. Candidate labels
are likewise originally observed source bins. All validation and test errors
are measured against the complete, approved OCHRE replay.

Weeks 1-4 precede validation, four purged rolling folds occur in week 5, and
week 6 is held out. The existing RF-to-p90 Outside-Hub controller provider is
issued separately without changing its fit, calibration, handoff, or p90 rule.
Its unavailable issues are recorded, and direct test comparisons use exactly
the origins where it and every candidate produced full eight-hour forecasts.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .evaluate_rest_of_home import read_15_minute_power
    from .evaluate_simulated_home_hub import (
        CV_DAY_OFFSETS, DAY, HORIZON_STEPS, TRAIN_WEEKS, WEEK,
        StudyRows, make_supervised, score_by_issue_day, score_predictions,
    )
    from .outside_hub_candidate_models import candidate_specs, make_candidate
    from .outside_hub_reference import (
        METHOD as REFERENCE_METHOD, SOURCE_START, issue_reference,
    )
    from ..single_home_hems.outside_hub_forecast_data import build_outside_hub_forecast_csv
except ImportError:  # direct script invocation from the repository root
    from evaluate_rest_of_home import read_15_minute_power
    from evaluate_simulated_home_hub import (
        CV_DAY_OFFSETS, DAY, HORIZON_STEPS, TRAIN_WEEKS, WEEK,
        StudyRows, make_supervised, score_by_issue_day, score_predictions,
    )
    from outside_hub_candidate_models import candidate_specs, make_candidate
    from outside_hub_reference import (
        METHOD as REFERENCE_METHOD, SOURCE_START, issue_reference,
    )
    from examples.single_home_hems.single_home_hems.outside_hub_forecast_data import (
        build_outside_hub_forecast_csv,
    )


BASELINE_DIR = (
    Path(__file__).resolve().parents[1]
    / "outputs/home_01/ev_home_hub_baseline_6wk"
)
DEFAULT_OUTPUT = BASELINE_DIR / "outside_hub_forecast_study"
SIMULATION_START = pd.Timestamp("2018-01-07 00:00:00")
EXPECTED_MINUTES = 6 * 7 * 24 * 60
EXPECTED_BLOCKS = 6 * 7 * 96
MINUTE = pd.Timedelta(minutes=1)


def _regular_index(index: pd.DatetimeIndex, *, start: pd.Timestamp,
                   periods: int, freq: str, label: str) -> None:
    expected = pd.date_range(start, periods=periods, freq=freq, name="timestamp")
    if (index.tz is not None or index.hasnans or index.has_duplicates
            or not index.equals(expected)):
        raise ValueError(f"{label} needs a complete, increasing six-week {freq} grid")


def load_replayed_truth(baseline_dir: Path) -> tuple[pd.Series, dict[str, float | int]]:
    """Derive the Outside-Hub target from completed total-home and Hub meters."""
    whole_path = baseline_dir / "data/home_01_standardized.csv"
    hub_path = baseline_dir / "data/home_hub_meter_minute.csv"
    fixed_path = baseline_dir / "home_hub_split_simulation_kw.csv"
    whole = pd.read_csv(
        whole_path, usecols=["timestamp", "total_electric_power_kw", "outside_hub_power_kw"],
        parse_dates=["timestamp"],
    )
    hub = pd.read_csv(
        hub_path, usecols=["timestamp", "home_hub_power_kw"],
        parse_dates=["timestamp"],
    )
    if len(whole) != EXPECTED_MINUTES or len(hub) != EXPECTED_MINUTES:
        raise ValueError("The baseline needs every evaluation minute")
    if not whole["timestamp"].equals(hub["timestamp"]):
        raise ValueError("Total-home and Home-Hub meters are not aligned")
    index = pd.DatetimeIndex(whole["timestamp"], name="timestamp")
    _regular_index(index, start=SIMULATION_START, periods=EXPECTED_MINUTES,
                   freq="min", label="Baseline meter")
    total = whole["total_electric_power_kw"].to_numpy(float)
    hub_power = hub["home_hub_power_kw"].to_numpy(float)
    outside = total - hub_power
    stored = whole["outside_hub_power_kw"].to_numpy(float)
    if (not np.isfinite(total).all() or not np.isfinite(hub_power).all()
            or not np.isfinite(outside).all() or (outside < -1e-8).any()):
        raise ValueError("Outside-Hub meter balance must be finite and nonnegative")
    balance = float(np.max(np.abs(outside - stored)))
    if balance > 1e-7:
        raise ValueError("Baseline Total minus Hub differs from stored Outside-Hub")
    fixed = pd.read_csv(
        fixed_path, usecols=["timestamp", "outside_hub_power_kw"],
        parse_dates=["timestamp"],
    ).set_index("timestamp")["outside_hub_power_kw"].reindex(index)
    fixed_balance = float(np.max(np.abs(outside - fixed.to_numpy(float))))
    if not np.isfinite(fixed_balance) or fixed_balance > 1e-7:
        raise ValueError("Fixed Outside-Hub replay differs from the completed meter")
    minute = pd.Series(np.maximum(outside, 0), index=index, name="outside_hub_kw")
    grouped = minute.resample("15min", closed="left", label="left")
    truth = grouped.mean()
    counts = grouped.count()
    _regular_index(truth.index, start=SIMULATION_START, periods=EXPECTED_BLOCKS,
                   freq="15min", label="Outside-Hub truth")
    if not counts.eq(15).all() or not np.isfinite(truth.to_numpy(float)).all():
        raise ValueError("Outside-Hub truth needs 15 complete minutes per block")
    return truth, {
        "minute_count": len(minute), "block_count": len(truth),
        "max_total_minus_hub_balance_kw": balance,
        "max_fixed_replay_balance_kw": fixed_balance,
    }


def observed_source_on_simulation_clock(
    source_path: Path, truth: pd.Series,
) -> tuple[pd.Series, dict[str, Any]]:
    """Map only originally observed source bins to the six-week OCHRE clock."""
    approved, observed, details = read_15_minute_power(source_path)
    shift = SIMULATION_START - SOURCE_START
    observed.index = observed.index + shift
    approved.index = approved.index + shift
    observed = (observed.reindex(truth.index) / 1000.0).rename("outside_hub_observed_kw")
    approved = approved.reindex(truth.index) / 1000.0
    valid = observed.notna()
    if not valid.any() or observed.loc[valid].lt(0).any():
        raise ValueError("No nonnegative originally observed Outside-Hub source bins")
    mismatch = float(np.max(np.abs(observed.loc[valid] - truth.loc[valid])))
    if mismatch > 1e-6:
        raise ValueError("Originally observed Outside-Hub source differs from OCHRE truth")
    details.update({
        "six_week_originally_observed_bins": int(valid.sum()),
        "six_week_approved_bins": int(approved.notna().sum()),
        "max_observed_vs_replay_kw": mismatch,
    })
    return observed, details


def supervised_rows(
    observed: pd.Series, truth: pd.Series,
) -> tuple[StudyRows, np.ndarray]:
    """Reuse the Hub benchmark's causal lag/calendar features for Outside-Hub."""
    if not observed.index.equals(truth.index):
        raise ValueError("Observed source and replay truth must share a 15-minute grid")
    # The shared feature builder names its input hub_non_ev_power_kw; here the
    # alias is internal only. All features derive from Outside-Hub observations.
    feature_input = pd.DataFrame({"hub_non_ev_power_kw": observed}, index=truth.index)
    observed_rows = make_supervised(feature_input)
    observed_labels = observed_rows.y.copy()
    replay_labels = truth.reindex(observed_rows.target).to_numpy(float)
    if not np.isfinite(replay_labels).all():
        raise ValueError("Every supervised target needs complete replay truth")
    rows = replace(observed_rows, y=replay_labels)
    return rows, observed_labels


def _prediction_rows(rows: StudyRows, mask: np.ndarray, forecast: np.ndarray,
                     method: str, weeks: int, split: str) -> pd.DataFrame:
    if len(forecast) != int(mask.sum()) or not np.isfinite(forecast).all():
        raise ValueError("Candidate forecast count or values are invalid")
    return pd.DataFrame({
        "method": method, "train_weeks": weeks, "split": split,
        "origin": rows.origin[mask], "target": rows.target[mask],
        "lead": rows.lead[mask], "forecast_kw": forecast,
        "actual_kw": rows.y[mask],
    })


def train_candidates(
    rows: StudyRows, observed_labels: np.ndarray, *, fast: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    """Choose parameters with week-5 rolling CV, then score week 6 once."""
    start = SIMULATION_START
    validation_start, test_start = start + 4 * WEEK, start + 5 * WEEK
    end = start + 6 * WEEK
    specs = candidate_specs()
    if fast:
        specs = {method: grid[:1] for method, grid in specs.items()}
    folds = CV_DAY_OFFSETS[:1] if fast else CV_DAY_OFFSETS
    observed_target = np.isfinite(observed_labels)
    cv_records: list[dict[str, Any]] = []
    selected: dict[str, dict[str, Any]] = {}
    predictions: list[pd.DataFrame] = []
    test = rows.mask_evaluation(test_start, end)
    if fast:
        test &= rows.origin.minute == 0
    for weeks in TRAIN_WEEKS:
        for method, grid in specs.items():
            columns = rows.feature_sets[
                "calendar" if method == "ridge_calendar" else "history"
            ]
            cv_maes: list[float] = []
            for number, params in enumerate(grid, 1):
                daily_maes: list[float] = []
                for day_offset in folds:
                    fold_start = validation_start + day_offset * DAY
                    fold_end = fold_start + DAY
                    train_start = fold_start - weeks * WEEK
                    fit = rows.mask_fit(train_start, fold_start) & observed_target
                    valid = rows.mask_evaluation(fold_start, fold_end)
                    if not fit.any() or not valid.any():
                        raise ValueError(f"Empty {weeks}-week fold at {fold_start}")
                    model = make_candidate(method, params)
                    model.fit(rows.x.loc[fit, list(columns)], observed_labels[fit])
                    forecast = np.maximum(
                        np.asarray(model.predict(rows.x.loc[valid, list(columns)]), float), 0.0,
                    )
                    if not np.isfinite(forecast).all():
                        raise ValueError(f"Nonfinite {method} validation forecast")
                    mae = float(np.mean(np.abs(forecast - rows.y[valid])))
                    daily_maes.append(mae)
                    cv_records.append({
                        "train_weeks": weeks, "method": method,
                        "candidate": number, "fold_start": fold_start,
                        "fold_end": fold_end, "fit_start": train_start,
                        "fit_end_exclusive": fold_start,
                        "fit_pairs": int(fit.sum()),
                        "validation_pairs": int(valid.sum()),
                        "mae_kw": mae,
                        "params": json.dumps(params, sort_keys=True),
                    })
                cv_maes.append(float(np.mean(daily_maes)))
            best = int(np.argmin(cv_maes))
            params = grid[best]
            selected[f"{weeks}w:{method}"] = {
                "params": params, "cv_mae_kw": cv_maes[best],
                "cv_grid_mae_kw": cv_maes,
            }
            fit = rows.mask_fit(test_start - weeks * WEEK, test_start) & observed_target
            model = make_candidate(method, params)
            model.fit(rows.x.loc[fit, list(columns)], observed_labels[fit])
            forecast = np.maximum(
                np.asarray(model.predict(rows.x.loc[test, list(columns)]), float), 0.0,
            )
            predictions.append(_prediction_rows(rows, test, forecast, method, weeks, "test"))
            print(f"{weeks}w {method}: CV {cv_maes[best]:.3f} kW; test issued", flush=True)
    return (pd.DataFrame(cv_records), pd.concat(predictions, ignore_index=True),
            selected)


def selected_families(selected: dict[str, dict[str, Any]]) -> dict[int, str]:
    """Choose a method for each training window using validation CV only."""
    choices: dict[int, str] = {}
    for weeks in TRAIN_WEEKS:
        candidates = [
            (float(result["cv_mae_kw"]), key.split(":", 1)[1])
            for key, result in selected.items() if key.startswith(f"{weeks}w:")
        ]
        if not candidates:
            raise ValueError(f"No validation models for {weeks} weeks")
        choices[weeks] = min(candidates)[1]
    return choices


def _score_row(scores: pd.DataFrame, *, method: str,
               weeks: int | None, hours: int = 8) -> pd.Series:
    chosen = scores.loc[
        scores["split"].eq("test") & scores["scope"].eq("through_horizon")
        & scores["horizon_hours"].eq(hours) & scores["method"].eq(method)
        & (scores["train_weeks"].isna() if weeks is None
           else scores["train_weeks"].eq(weeks))
    ]
    if len(chosen) != 1:
        raise ValueError(f"Expected one score for {method}, {weeks}w, {hours}h")
    return chosen.iloc[0]


def _fmt(value: Any, digits: int = 3) -> str:
    number = float(value)
    return f"{number:.{digits}f}" if np.isfinite(number) else "n/a"


def write_report(
    output_dir: Path, *, full_scores: pd.DataFrame, matched_scores: pd.DataFrame,
    selected: dict[str, dict[str, Any]], coverage: pd.DataFrame,
    threshold_kw: float, observed: pd.Series, truth: pd.Series,
) -> None:
    """Write validation-locked results with the reference coverage explicit."""
    choices = selected_families(selected)
    heldout = coverage.set_index("split").loc["test"]
    lines = [
        "# Six-week Outside-Hub forecast comparison", "",
        "The target is the 15-minute mean Outside-Hub load, derived from the "
        "completed OCHRE total-home meter minus the Home-Hub meter. It is a "
        "different target from the non-EV Hub study. Candidate features and "
        "fit labels use only originally observed Outside-Hub source bins; "
        "the OCHRE replay supplies complete validation/test truth.", "",
        "Weeks 1–4 supply development history. Four rolling, purged day folds "
        "in week 5 select parameters and model families; week 6 is held out. "
        "The existing RF→p90 reference is fitted once before week 1 using its "
        "original 20-day source history and Aug 2–21 peak threshold. Its clock "
        "p90 history updates from completed earlier source days. The original "
        "two-hour handoff and causal short-gap inference rule are preserved.", "",
        "## Validation-selected models on held-out week 6", "",
        "| Fit-label window | Week-5 choice | CV MAE (kW) | Full 8h MAE (kW) | "
        "Matched 8h MAE (kW) | Matched bias (kW) | Matched peak recall | "
        "Peak shortfall (kW) |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for weeks in TRAIN_WEEKS:
        method = choices[weeks]
        cv = selected[f"{weeks}w:{method}"]["cv_mae_kw"]
        full = _score_row(full_scores, method=method, weeks=weeks)
        matched = _score_row(matched_scores, method=method, weeks=weeks)
        lines.append(
            f"| {weeks} weeks | {method} | {_fmt(cv)} | {_fmt(full['mae_kw'])} | "
            f"{_fmt(matched['mae_kw'])} | {_fmt(matched['bias_kw'])} | "
            f"{_fmt(matched['peak_recall'])} | "
            f"{_fmt(matched['actual_peak_mean_shortfall_kw'])} |"
        )
    reference = _score_row(matched_scores, method=REFERENCE_METHOD, weeks=None)
    lines += [
        "", "The unchanged reference, on the **same matched held-out issues**, "
        f"has eight-hour MAE {_fmt(reference['mae_kw'])} kW, bias "
        f"{_fmt(reference['bias_kw'])} kW, and peak recall "
        f"{_fmt(reference['peak_recall'])}, with mean peak shortfall "
        f"{_fmt(reference['actual_peak_mean_shortfall_kw'])} kW. "
        "Peak shortfall averages actual minus forecast when positive, among "
        "above-threshold actual forecast-target pairs. It issued full eight-hour forecasts "
        f"at {int(heldout['issued_origins'])}/{int(heldout['requested_origins'])} "
        "test origins. Missing issues are listed in "
        "[reference_unavailable_origins.csv](reference_unavailable_origins.csv).", "",
        "The full scores cover all candidate test origins; the matched scores "
        "restrict every method to origins where the deployed reference issued. "
        "Use the matched columns for direct comparisons. Candidate fit windows "
        "limit fit origins and labels to the trailing 2, 3, or 4 weeks, while "
        "causally observed lag features can reach farther back. The reference "
        "keeps its original pre-week-1 RF fit, whereas candidate models are "
        "refitted immediately before week 6. Thus this compares the existing "
        "deployed forecast policy with newly trained candidates, not model "
        "families at the same fit vintage.", "",
        "## HGB and other modern tree models", "",
        "The candidate set includes HGB, LightGBM, XGBoost, and CatBoost, "
        "plus random forest and Ridge baselines. Every history model receives "
        "the same completed-observation lag and calendar features and the "
        "same chronological folds. Ridge-calendar omits lagged load inputs.", "",
        "| Method | 2w matched 8h MAE | 3w matched 8h MAE | 4w matched 8h MAE |",
        "| --- | ---: | ---: | ---: |",
    ]
    for method in candidate_specs():
        values = [
            _fmt(_score_row(matched_scores, method=method, weeks=weeks)["mae_kw"])
            for weeks in TRAIN_WEEKS
        ]
        lines.append(f"| {method} | {' | '.join(values)} |")
    test_observed = observed.loc[
        (observed.index >= SIMULATION_START + 5 * WEEK)
        & (observed.index < SIMULATION_START + 6 * WEEK)
    ]
    test_truth = truth.loc[
        (truth.index >= SIMULATION_START + 5 * WEEK)
        & (truth.index < SIMULATION_START + 6 * WEEK)
    ]
    peak_truth = test_truth.loc[test_truth.ge(threshold_kw)]
    lines += [
        "", "## Peak and coverage interpretation", "",
        f"The peak threshold is {_fmt(threshold_kw)} kW: the existing "
        "Outside-Hub reference's 95th percentile of originally observed "
        "August 2–21 bins, fixed before this six-week study. Week-6 source "
        f"coverage is {int(test_observed.notna().sum())}/{len(test_observed)} "
        "originally observed 15-minute blocks. The test contains "
        f"{len(peak_truth)} distinct above-threshold intervals across "
        f"{peak_truth.index.normalize().nunique()} days. Peak recall in the "
        "tables counts forecast-target pairs, so overlapping eight-hour "
        "issues repeat these intervals and do not create independent events. "
        "A conservative p90 forecast "
        "may trade larger MAE and upward bias for greater peak coverage; these "
        "metrics alone do not establish a controller replacement. The forecast "
        "is evaluated off-line and is not wired into EV control.", "",
        "## Artifacts", "",
        "[Full scores](horizon_scores.csv), "
        "[matched scores](matched_horizon_scores.csv), "
        "[lead scores](matched_lead_scores.csv), "
        "[rolling week-5 folds](rolling_cv.csv), "
        "[origin-level predictions](predictions.csv.gz), and "
        "[issue-day scores](matched_issue_day_scores.csv). "
        "Overlapping forecast horizons produce correlated issue-target pairs; "
        "day scores retain useful units for paired comparisons.", "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run(
    *, baseline_dir: Path = BASELINE_DIR, output_dir: Path = DEFAULT_OUTPUT,
    source_csv: Path | None = None, fast: bool = False,
) -> None:
    """Build source once, validate causal data, fit, issue, score, and save."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if source_csv is None:
        source_path = build_outside_hub_forecast_csv(
            output_dir / "outside_hub_forecast_source.csv"
        )
    else:
        source_path = source_csv.expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
    truth, truth_audit = load_replayed_truth(baseline_dir)
    observed, source_audit = observed_source_on_simulation_clock(source_path, truth)
    rows, observed_labels = supervised_rows(observed, truth)
    cv, candidates, selected = train_candidates(rows, observed_labels, fast=fast)
    validation_start = SIMULATION_START + 4 * WEEK
    test_start = SIMULATION_START + 5 * WEEK
    end = SIMULATION_START + 6 * WEEK
    requested: dict[str, pd.DatetimeIndex] = {}
    for split, start, stop in (
        ("validation", validation_start, test_start),
        ("test", test_start, end),
    ):
        mask = rows.mask_evaluation(start, stop)
        if fast:
            mask &= rows.origin.minute == 0
        requested[split] = pd.DatetimeIndex(rows.origin[mask].unique())
    ref_origins = requested["validation"].append(requested["test"])
    reference, unavailable, reference_metadata = issue_reference(
        source_path=source_path, origins=ref_origins,
        actual_at_target=truth, simulation_start=SIMULATION_START,
        source_start=SOURCE_START,
    )
    if reference.empty:
        raise ValueError("The existing Outside-Hub reference issued no forecasts")
    reference["split"] = np.where(reference["origin"] < test_start, "validation", "test")
    if not unavailable.empty:
        unavailable["split"] = np.where(
            unavailable["origin"] < test_start, "validation", "test"
        )
    coverage_records = []
    for split, origins in requested.items():
        issued = reference.loc[reference["split"].eq(split), "origin"].nunique()
        skipped = int(unavailable["split"].eq(split).sum()) if not unavailable.empty else 0
        if issued + skipped != len(origins):
            raise AssertionError(f"Reference coverage does not sum to requested {split} issues")
        coverage_records.append({
            "split": split, "requested_origins": len(origins),
            "issued_origins": issued, "unavailable_origins": skipped,
            "coverage_fraction": issued / len(origins),
        })
    coverage = pd.DataFrame(coverage_records)
    predictions = pd.concat([candidates, reference], ignore_index=True)
    threshold_kw = float(reference_metadata["peak_threshold_w"]) / 1000.0
    full_scores, full_leads = score_predictions(predictions, threshold_kw)
    matched_origins = pd.DatetimeIndex(
        reference.loc[reference["split"].eq("test"), "origin"].unique()
    )
    if matched_origins.empty:
        raise ValueError("The existing reference has no held-out matching origins")
    matched = predictions.loc[
        predictions["split"].eq("test") & predictions["origin"].isin(matched_origins)
    ].copy()
    expected_pairs = len(matched_origins) * HORIZON_STEPS
    for (method, weeks), group in matched.groupby(["method", "train_weeks"], dropna=False):
        if len(group) != expected_pairs:
            raise AssertionError(f"{method}/{weeks} lacks the common matched test pairs")
    matched_scores, matched_leads = score_predictions(matched, threshold_kw)
    with gzip.open(output_dir / "predictions.csv.gz", "wt", encoding="utf-8") as stream:
        predictions.to_csv(stream, index=False)
    pd.DataFrame({
        "timestamp": truth.index, "outside_hub_replayed_kw": truth.to_numpy(float),
        "outside_hub_originally_observed_kw": observed.to_numpy(float),
        "source_observed": observed.notna().to_numpy(),
    }).to_csv(output_dir / "outside_hub_15min.csv", index=False)
    cv.to_csv(output_dir / "rolling_cv.csv", index=False)
    coverage.to_csv(output_dir / "reference_coverage.csv", index=False)
    unavailable.to_csv(output_dir / "reference_unavailable_origins.csv", index=False)
    full_scores.to_csv(output_dir / "horizon_scores.csv", index=False)
    full_leads.to_csv(output_dir / "lead_scores.csv", index=False)
    matched_scores.to_csv(output_dir / "matched_horizon_scores.csv", index=False)
    matched_leads.to_csv(output_dir / "matched_lead_scores.csv", index=False)
    score_by_issue_day(predictions, threshold_kw).to_csv(
        output_dir / "issue_day_scores.csv", index=False
    )
    score_by_issue_day(matched, threshold_kw).to_csv(
        output_dir / "matched_issue_day_scores.csv", index=False
    )
    metadata = {
        "baseline_directory": str(baseline_dir.resolve()),
        "source_csv": str(source_path.resolve()),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "simulation_start": SIMULATION_START.isoformat(),
        "source_start": SOURCE_START.isoformat(),
        "validation_start": validation_start.isoformat(),
        "test_start": test_start.isoformat(),
        "test_end": end.isoformat(),
        "interval_minutes": 15, "horizon_steps": HORIZON_STEPS,
        "fit_origin_stride_minutes": 60,
        "evaluation_origin_stride_minutes": 60 if fast else 15,
        "training_weeks": TRAIN_WEEKS,
        "cv_fold_day_offsets_in_week5": CV_DAY_OFFSETS[:1] if fast else CV_DAY_OFFSETS,
        "cv_rule": "trailing N-week fit labels; full-horizon label purge; daily rolling folds",
        "candidate_input_rule": (
            "Originally observed Outside-Hub 15-minute source bins strictly before "
            "each issue; missing values become zero with availability/count flags; "
            "calendar/lead are known at issue"
        ),
        "candidate_fit_label_rule": "originally observed source bins only",
        "scoring_truth_rule": "complete approved OCHRE Outside-Hub replay",
        "reference_method": REFERENCE_METHOD,
        "reference_provider_metadata": reference_metadata,
        "reference_coverage": coverage.to_dict(orient="records"),
        "matched_test_origins": len(matched_origins),
        "peak_threshold_kw": threshold_kw,
        "peak_threshold_rule": "existing reference's originally observed Aug 2–21 p95",
        "selected_validation_params": selected,
        "selected_validation_methods": selected_families(selected),
        "truth_audit": truth_audit,
        "source_audit": source_audit,
        "package_versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scikit-learn", "lightgbm", "xgboost", "catboost")
        },
        "fast": fast,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    write_report(
        output_dir, full_scores=full_scores, matched_scores=matched_scores,
        selected=selected, coverage=coverage, threshold_kw=threshold_kw,
        observed=observed, truth=truth,
    )
    print(f"Saved Outside-Hub study to {output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=BASELINE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-csv", type=Path,
                        help="Existing quality-flagged Outside-Hub source; default builds it")
    parser.add_argument("--fast", action="store_true",
                        help="One validation fold/setting and hourly test origins for smoke runs")
    args = parser.parse_args()
    run(baseline_dir=args.baseline_dir, output_dir=args.output_dir,
        source_csv=args.source_csv, fast=args.fast)


if __name__ == "__main__":
    main()
