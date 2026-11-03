"""Direct aggregate-meter QCP follow-up for the EV-only Outside-Hub+ER target.

The QCP source is the simulated measured aggregate total-home minus corrected
Home Hub, masked to the source's approved-quality minutes. This is the meter
the EV-only controller can observe; the private electric-resistance channel is
used only after prediction to audit QCP fidelity. Each approved 15-minute bin
inherits the original source's conservative closing-observation readiness, and
an hourly QCP input is usable only after all four bins and their readiness.

The two new RFs add the approved signed symmetric heating QCP feature to the
established target-history RF, with and without the latest observed ambient
temperature. Saved plain and past-weather RFs are exact matched references.
All RF labels remain originally observed target bins only.
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

from .evaluate_outside_hub import (
    BASELINE_DIR, CV_DAY_OFFSETS, DAY, EXPECTED_MINUTES, SIMULATION_START,
    supervised_rows,
)
from .evaluate_outside_hub_with_er import (
    CORRECTED_BASELINE_DIR, DEFAULT_OUTPUT as CORE_DIR, ER_COLUMN, _sha256,
    load_composite_target,
)
from .evaluate_outside_hub_with_er_rf_qcp import (
    DEFAULT_OUTPUT as HUB_QCP_DIR, KEY, LEAVES, WEATHER_COLUMNS,
    _candidate_forecast, _common_origins, _coverage_row, _forecast_rows,
    _read_hub_and_weather, _saved_rf, _weather_matrix,
)
from .evaluate_rest_of_home import read_15_minute_power
from .evaluate_simulated_home_hub import (
    HORIZON_STEPS, INTERVAL, QCP_COLUMNS, TRAIN_WEEKS, WEEK,
    _fold_qcp_matrices, score_by_issue_day, score_predictions,
)
from .outside_hub_candidate_models import make_candidate
from .outside_hub_reference import SOURCE_START
from .qcp_heating_feature import (
    HOUR, LOCAL_TZ, issue_qcp_heating_feature, try_fit_hour_pooled_qcp,
)
from examples.single_home_hems.single_home_hems.rest_of_home_forecast import (
    _approved_bin_ready_times,
)


DEFAULT_OUTPUT = CORE_DIR / "supplemental/rf_history_direct_outside_qcp"
DIRECT_METHODS = (
    "rf_history_direct_outside_qcp",
    "rf_history_weather_direct_outside_qcp",
)
REFERENCE_METHODS = ("rf_history", "rf_history_weather_lag")


def _read_direct_qcp_source(
    *, core_dir: Path, corrected_baseline_dir: Path,
    truth: pd.Series, observed: pd.Series, weather: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build ready-aware QCP hours from public aggregate, not private ER power."""
    source_path = core_dir / "outside_hub_with_resistance_forecast_source.csv"
    hybrid_metadata_path = core_dir / "hybrid_metadata.json"
    hybrid_metadata = json.loads(hybrid_metadata_path.read_text(encoding="utf-8"))
    source_sha = _sha256(source_path)
    if source_sha != hybrid_metadata["source_audit"]["source_sha256"]:
        raise ValueError("Approved-quality source changed from the saved hybrid study")
    source_quality = pd.read_csv(
        source_path, usecols=["Time", "Quality"], parse_dates=["Time"]
    )
    source_times = pd.DatetimeIndex(source_quality["Time"])
    expected_source = pd.date_range(
        SOURCE_START + pd.Timedelta(minutes=1), periods=EXPECTED_MINUTES, freq="min"
    )
    if not source_times.equals(expected_source):
        raise ValueError("Composite quality source lacks the exact six-week minute grid")
    quality = source_quality["Quality"].astype(str).to_numpy()
    valid_minute = np.isin(quality, ("observed", "interpolated"))
    observed_minute = quality == "observed"

    total_path = BASELINE_DIR / "data/home_01_standardized.csv"
    hub_path = corrected_baseline_dir / "data/home_hub_meter_minute.csv"
    total = pd.read_csv(total_path, usecols=["timestamp", "total_electric_power_kw"],
                        parse_dates=["timestamp"])
    hub = pd.read_csv(hub_path, usecols=["timestamp", "home_hub_power_kw"],
                      parse_dates=["timestamp"])
    minute_index = pd.date_range(
        SIMULATION_START, periods=EXPECTED_MINUTES, freq="min", name="timestamp"
    )
    if (len(total) != EXPECTED_MINUTES or len(hub) != EXPECTED_MINUTES
            or not pd.DatetimeIndex(total["timestamp"]).equals(minute_index)
            or not pd.DatetimeIndex(hub["timestamp"]).equals(minute_index)):
        raise ValueError("Total-home and corrected Hub meters must share the minute grid")
    aggregate_kw = (
        total["total_electric_power_kw"].to_numpy(float)
        - hub["home_hub_power_kw"].to_numpy(float)
    )
    if not np.isfinite(aggregate_kw).all() or np.min(aggregate_kw) < -1e-7:
        raise ValueError("Direct QCP aggregate meter is invalid")
    aggregate_kw = np.maximum(aggregate_kw, 0.0)
    approved_minute = pd.Series(
        np.where(valid_minute, aggregate_kw, np.nan), index=minute_index
    )
    originally_observed_minute = pd.Series(
        np.where(observed_minute, aggregate_kw, np.nan), index=minute_index
    )
    approved_groups = approved_minute.resample("15min", closed="left", label="left")
    observed_groups = originally_observed_minute.resample(
        "15min", closed="left", label="left"
    )
    approved_15m = approved_groups.mean().where(approved_groups.count().eq(15))
    observed_15m = observed_groups.mean().where(observed_groups.count().eq(15))
    if (not approved_15m.index.equals(truth.index)
            or not observed_15m.index.equals(observed.index)
            or not observed_15m.notna().equals(observed.notna())
            or not np.allclose(observed_15m, observed, rtol=0, atol=1e-6,
                               equal_nan=True)):
        raise ValueError("Aggregate meter and original observed target mask disagree")

    # Read the old composite Value only for independent provenance validation
    # and the existing readiness helper. It never enters an RF/QCP input.
    source_approved_w, source_observed_w, _ = read_15_minute_power(source_path)
    shift = SIMULATION_START - SOURCE_START
    saved_approved = source_approved_w.copy()
    saved_approved.index += shift
    saved_approved = saved_approved.reindex(truth.index) / 1000.0
    if (not saved_approved.notna().equals(approved_15m.notna())
            or not np.allclose(saved_approved, approved_15m, rtol=0, atol=1e-6,
                               equal_nan=True)):
        raise ValueError("Measured aggregate disagrees with the saved approved source")
    ready_source = _approved_bin_ready_times(
        source_path, source_approved_w, source_observed_w
    )
    ready_sim = ready_source.copy()
    ready_sim.index += shift
    ready_sim = (ready_sim + shift).reindex(truth.index)
    if (not ready_sim.notna().equals(approved_15m.notna())
            or (ready_sim.loc[approved_15m.notna()]
                < approved_15m.index[approved_15m.notna()] + INTERVAL).any()):
        raise ValueError("Approved QCP bin readiness is missing or precedes bin end")
    if not (
        ready_sim.loc[observed_15m.notna()].to_numpy()
        == (observed_15m.index[observed_15m.notna()] + INTERVAL).to_numpy()
    ).all():
        raise ValueError("Originally observed aggregate bins must be ready at their ends")

    local_index = truth.index.tz_localize(LOCAL_TZ)
    ready_utc = pd.to_datetime(ready_sim).dt.tz_localize(LOCAL_TZ).dt.tz_convert("UTC")
    blocks = pd.DataFrame({
        "total_kw": approved_15m.to_numpy(float),
        "temperature_c": weather.reindex(truth.index).to_numpy(float),
        "ready_at_utc": ready_utc.to_numpy(),
        "source_observed": observed_15m.notna().to_numpy(),
    }, index=local_index)
    grouped = blocks.groupby(blocks.index.floor("h"), sort=True)
    hourly = grouped.agg(
        total_energy_kwh=("total_kw", "mean"),
        temperature_c=("temperature_c", "mean"),
        block_count=("total_kw", "size"),
        valid_count=("total_kw", "count"),
        ready_at_utc=("ready_at_utc", "max"),
        originally_observed_bins=("source_observed", "sum"),
    )
    hourly = hourly.loc[
        hourly["block_count"].eq(4) & hourly["valid_count"].eq(4)
        & hourly["temperature_c"].notna() & hourly["ready_at_utc"].notna()
    ].copy()
    hourly.index.name = "hour_start_local"
    hourly["hour_end_utc"] = (hourly.index + HOUR).tz_convert("UTC")
    hourly["ready_at_utc"] = pd.to_datetime(hourly["ready_at_utc"], utc=True)
    hourly["ready_at_utc"] = hourly[["ready_at_utc", "hour_end_utc"]].max(axis=1)
    if (hourly["ready_at_utc"] < hourly["hour_end_utc"]).any():
        raise ValueError("Direct QCP hour ready before the final 15-minute bin")
    for hour, row in hourly.iterrows():
        starts = pd.date_range(hour.tz_localize(None), periods=4, freq=INTERVAL)
        bin_ready = ready_sim.reindex(starts)
        if bin_ready.isna().any():
            raise ValueError("A complete direct-QCP hour lacks a bin readiness")
        expected_ready = max(
            row["hour_end_utc"],
            bin_ready.max().tz_localize(LOCAL_TZ).tz_convert("UTC"),
        )
        if row["ready_at_utc"] != expected_ready:
            raise ValueError("Direct-QCP hourly readiness does not equal four-bin maximum")
    if (hourly["total_energy_kwh"].lt(-1e-7).any()
            or not np.isfinite(hourly[["total_energy_kwh", "temperature_c"]]
                               .to_numpy(float)).all()):
        raise ValueError("Direct QCP hourly aggregate/weather is invalid")
    hourly = hourly.drop(columns=["block_count", "valid_count"])
    audit = {
        "composite_quality_source_sha256": source_sha,
        "hybrid_metadata_sha256": _sha256(hybrid_metadata_path),
        "total_home_meter_sha256": _sha256(total_path),
        "corrected_hub_meter_sha256": _sha256(hub_path),
        "aggregate_equation": "simulated total-home meter minus corrected Home Hub meter",
        "quality_rule": "read aggregate only at source observed/interpolated quality minutes",
        "readiness_rule": "approved source closing-observed-endpoint readiness; hour=max(hour end, four bin ready times)",
        "approved_15min_bins": int(approved_15m.notna().sum()),
        "originally_observed_15min_bins": int(observed_15m.notna().sum()),
        "complete_ready_qcp_hours": len(hourly),
        "approved_aggregate_vs_saved_source_max_kw": float(
            np.nanmax(np.abs(approved_15m - saved_approved))
        ),
        "approved_bins_delayed_past_end": int((
            ready_sim.loc[approved_15m.notna()].to_numpy()
            > (approved_15m.index[approved_15m.notna()] + INTERVAL).to_numpy()
        ).sum()),
        "qcp_hours_delayed_past_end": int((
            hourly["ready_at_utc"] > hourly["hour_end_utc"]
        ).sum()),
    }
    return hourly, audit


def _saved_weather_rf(hub_qcp_dir: Path, rows, test_mask: np.ndarray,
                      target_audit: dict[str, Any],
                      weather_sha256: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = hub_qcp_dir / "predictions.csv.gz"
    metadata_path = hub_qcp_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (metadata.get("target") != target_audit["target"]
            or metadata.get("target_observation_mask")
            != "originally observed Outside-Hub source bins only"
            or metadata.get("realized_weather_15min_sha256") != weather_sha256
            or metadata.get("training_weeks") != list(TRAIN_WEEKS)
            or metadata.get("cv_fold_day_offsets_in_week5") != list(CV_DAY_OFFSETS)):
        raise ValueError("Saved weather RF benchmark has different target or split provenance")
    weather = pd.read_csv(path, parse_dates=["origin", "target"])
    weather = weather.loc[
        weather["method"].eq("rf_history_weather_lag")
        & weather["split"].eq("test")
    ].sort_values(KEY).reset_index(drop=True)
    expected = pd.concat([
        _forecast_rows(rows, test_mask, np.full(int(test_mask.sum()), np.nan),
                       "rf_history_weather_lag", weeks)
        for weeks in TRAIN_WEEKS
    ], ignore_index=True).sort_values(KEY).reset_index(drop=True)
    if (len(weather) != len(expected) or not weather[KEY].equals(expected[KEY])
            or not np.allclose(weather["actual_kw"], expected["actual_kw"],
                               rtol=0, atol=1e-8)
            or not np.isfinite(weather["forecast_kw"]).all()):
        raise ValueError("Saved weather RF does not match the direct QCP test grid/truth")
    return weather, {
        "saved_weather_rf_predictions_sha256": _sha256(path),
        "saved_weather_rf_metadata_sha256": _sha256(metadata_path),
        "saved_weather_rf_selected_params": {
            f"{weeks}w": metadata["selected_validation_params"][f"{weeks}w:rf_history_weather_lag"]
            for weeks in TRAIN_WEEKS
        },
    }


def _direct_matrices(method: str, fit_mask: np.ndarray, valid_mask: np.ndarray,
                     qtrain: pd.DataFrame, qvalid: pd.DataFrame,
                     weather_x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if method == DIRECT_METHODS[0]:
        return qtrain, qvalid
    if method == DIRECT_METHODS[1]:
        return (
            qtrain.join(weather_x.loc[fit_mask, list(WEATHER_COLUMNS)],
                        validate="one_to_one"),
            qvalid.join(weather_x.loc[valid_mask, list(WEATHER_COLUMNS)],
                        validate="one_to_one"),
        )
    raise ValueError(f"Unknown direct-QCP model {method}")


def _diagnose_completed_hours(hourly: pd.DataFrame,
                              test_start: pd.Timestamp, test_end: pd.Timestamp,
                              *, source_model_start: pd.Timestamp,
                              model, train_weeks: int) -> pd.DataFrame:
    """Retrospective signed QCP-vs-hidden-ER check; never an RF input."""
    raw_path = BASELINE_DIR / "data/raw/home_01.csv"
    raw = pd.read_csv(raw_path, usecols=["Time", ER_COLUMN], parse_dates=["Time"])
    raw = raw.set_index("Time")[ER_COLUMN].astype(float)
    expected = pd.date_range(SIMULATION_START, periods=EXPECTED_MINUTES, freq="min")
    if raw.index.has_duplicates or not raw.index.is_monotonic_increasing:
        raise ValueError("Hidden ER diagnostic timestamps are invalid")
    raw = raw.reindex(expected)
    if not np.isfinite(raw.to_numpy(float)).all():
        raise ValueError("Hidden ER diagnostic label must span the exact minute grid")
    er_hourly = raw.resample("h", closed="left", label="left").mean()
    records = []
    local_start = test_start.tz_localize(LOCAL_TZ)
    local_end = test_end.tz_localize(LOCAL_TZ)
    for hour, row in hourly.loc[
        (hourly.index >= local_start) & (hourly.index < local_end)
    ].iterrows():
        ready_utc = max(row["hour_end_utc"], row["ready_at_utc"])
        if ready_utc > local_end.tz_convert("UTC"):
            continue
        issued = issue_qcp_heating_feature(
            model, hourly.loc[:hour], ready_utc.tz_convert(LOCAL_TZ)
        )
        if issued.source.source_hour_local != hour:
            raise ValueError("Diagnostic QCP source hour differs from evaluated hour")
        estimate = float(issued.heating_feature_kw)
        actual = float(er_hourly.loc[hour.tz_localize(None)])
        records.append({
            "train_weeks": train_weeks,
            "hour_start": hour.tz_localize(None),
            "hour_end": row["hour_end_utc"].tz_convert(LOCAL_TZ).tz_localize(None),
            "ready_at": ready_utc.tz_convert(LOCAL_TZ).tz_localize(None),
            "ready_delay_minutes": float(
                (ready_utc - row["hour_end_utc"]).total_seconds() / 60
            ),
            "originally_observed_bins": int(row["originally_observed_bins"]),
            "qcp_signed_heating_kw": estimate,
            "hidden_er_actual_kw": actual,
            "error_kw": estimate - actual,
            "actual_er_on": bool(actual > 1e-9),
            "qcp_estimate_positive": bool(estimate > 0.0),
            "qcp_fit_start": source_model_start,
            "qcp_fit_cutoff_exclusive": test_start,
        })
    return pd.DataFrame(records)


def _diagnostic_summary(diagnostic: pd.DataFrame) -> pd.DataFrame:
    summary = []
    for weeks, group in diagnostic.groupby("train_weeks", sort=True):
        error = group["error_kw"].to_numpy(float)
        actual = group["hidden_er_actual_kw"].to_numpy(float)
        estimate = group["qcp_signed_heating_kw"].to_numpy(float)
        actual_on = group["actual_er_on"].to_numpy(bool)
        estimate_on = group["qcp_estimate_positive"].to_numpy(bool)
        summary.append({
            "train_weeks": weeks, "eligible_test_hours": len(group),
            "test_hours": 7 * 24,
            "coverage_fraction": len(group) / (7 * 24),
            "mae_kw": float(np.abs(error).mean()),
            "bias_kw": float(error.mean()),
            "correlation": float(np.corrcoef(estimate, actual)[0, 1]),
            "actual_er_on_hours": int(actual_on.sum()),
            "qcp_positive_hours": int(estimate_on.sum()),
            "on_recall": float(estimate_on[actual_on].mean()) if actual_on.any() else np.nan,
            "on_precision": float(actual_on[estimate_on].mean()) if estimate_on.any() else np.nan,
            "delayed_ready_hours": int(group["ready_delay_minutes"].gt(0).sum()),
            "max_ready_delay_minutes": float(group["ready_delay_minutes"].max()),
        })
    return pd.DataFrame(summary)


def _issue_freshness(hourly: pd.DataFrame, model,
                     origins: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Audit age of the latest completed ready QCP hour at every test issue."""
    records = []
    for origin in origins:
        issued = issue_qcp_heating_feature(model, hourly, origin)
        source = issued.source
        origin_utc = origin.tz_localize(LOCAL_TZ).tz_convert("UTC")
        source_end_utc = source.source_hour_local.tz_convert("UTC") + HOUR
        ready_utc = source.source_ready_at_utc
        if source_end_utc > origin_utc or ready_utc > origin_utc:
            raise ValueError("QCP test issue used an incomplete or unready source hour")
        records.append({
            "origin": origin,
            "source_hour_start": source.source_hour_local.tz_localize(None),
            "source_hour_end": source_end_utc.tz_convert(LOCAL_TZ).tz_localize(None),
            "source_ready_at": ready_utc.tz_convert(LOCAL_TZ).tz_localize(None),
            "age_minutes": float((origin_utc - source_end_utc).total_seconds() / 60),
            "ready_delay_minutes": float((ready_utc - source_end_utc).total_seconds() / 60),
        })
    frame = pd.DataFrame(records)
    if len(frame) != len(origins) or frame["origin"].duplicated().any():
        raise ValueError("QCP freshness audit lacks test issues")
    age = frame["age_minutes"].to_numpy(float)
    summary = {
        "test_issues": len(frame),
        "median_age_minutes": float(np.median(age)),
        "p95_age_minutes": float(np.quantile(age, 0.95)),
        "max_age_minutes": float(np.max(age)),
        "issues_older_than_one_hour": int(np.sum(age > 60)),
        "max_source_ready_delay_minutes": float(frame["ready_delay_minutes"].max()),
    }
    return frame, summary


def run(*, core_dir: Path = CORE_DIR,
        corrected_baseline_dir: Path = CORRECTED_BASELINE_DIR,
        hub_qcp_dir: Path = HUB_QCP_DIR,
        output_dir: Path = DEFAULT_OUTPUT) -> None:
    core_dir = core_dir.expanduser().resolve()
    corrected_baseline_dir = corrected_baseline_dir.expanduser().resolve()
    hub_qcp_dir = hub_qcp_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    truth, observed, target_audit = load_composite_target(
        corrected_baseline_dir=corrected_baseline_dir,
    )
    rows, observed_labels = supervised_rows(observed, truth)
    public, public_hashes = _read_hub_and_weather(corrected_baseline_dir, truth.index)
    weather = public["ambient_temperature_c"]
    hourly, source_audit = _read_direct_qcp_source(
        core_dir=core_dir, corrected_baseline_dir=corrected_baseline_dir,
        truth=truth, observed=observed, weather=weather,
    )
    weather_x = _weather_matrix(rows, weather)
    validation_start = SIMULATION_START + 4 * WEEK
    test_start, test_end = SIMULATION_START + 5 * WEEK, SIMULATION_START + 6 * WEEK
    test_mask = rows.mask_evaluation(test_start, test_end)
    observed_target = np.isfinite(observed_labels)
    if int(test_mask.sum()) != 641 * HORIZON_STEPS:
        raise ValueError("Direct-QCP test needs exactly 641 complete eight-hour issues")
    baseline, baseline_provenance = _saved_rf(core_dir, rows, test_mask, target_audit)
    weather_baseline, weather_provenance = _saved_weather_rf(
        hub_qcp_dir, rows, test_mask, target_audit,
        public_hashes["realized_weather_15min_sha256"],
    )
    common_origins, hybrid_provenance = _common_origins(core_dir)
    if not common_origins.isin(pd.DatetimeIndex(rows.origin[test_mask].unique())).all():
        raise ValueError("Hybrid matched issue grid changed")

    cv_records: list[dict[str, Any]] = []
    coverage_records: list[dict[str, Any]] = []
    selected: dict[str, dict[str, Any]] = {}
    predictions = [baseline, weather_baseline]
    final_qcp_models: dict[int, tuple[Any, pd.Timestamp]] = {}
    freshness: pd.DataFrame | None = None
    freshness_summary: dict[str, Any] | None = None
    for weeks in TRAIN_WEEKS:
        fold_cache: dict[int, dict[str, Any]] = {}
        for offset in CV_DAY_OFFSETS:
            fold_start = validation_start + offset * DAY
            fold_end = fold_start + DAY
            train_start = fold_start - weeks * WEEK
            fit = rows.mask_fit(train_start, fold_start) & observed_target
            valid = rows.mask_evaluation(fold_start, fold_end)
            if not fit.any() or not valid.any():
                raise ValueError(f"Empty direct-QCP {weeks}w fold at {fold_start}")
            qtrain, qvalid, info = _fold_qcp_matrices(
                hourly, rows, fit, valid, train_start, fold_start
            )
            if not info["evaluation_fit"]["fit_available"]:
                raise ValueError(f"Direct QCP unavailable for {weeks}w fold {offset}")
            coverage_records.append(_coverage_row(
                weeks, f"validation_day_{offset}", train_start, fold_start,
                info, qtrain, qvalid,
            ))
            fold_cache[offset] = {
                "fit": fit, "valid": valid, "qtrain": qtrain, "qvalid": qvalid,
                "fold_start": fold_start, "fold_end": fold_end,
                "train_start": train_start,
            }
            print(f"Prepared direct QCP {weeks}w fold {offset}", flush=True)

        final_start = test_start - weeks * WEEK
        final_fit = rows.mask_fit(final_start, test_start) & observed_target
        qtrain_final, qtest, final_info = _fold_qcp_matrices(
            hourly, rows, final_fit, test_mask, final_start, test_start
        )
        if not final_info["evaluation_fit"]["fit_available"]:
            raise ValueError(f"Direct QCP unavailable for {weeks}w test fit")
        coverage_records.append(_coverage_row(
            weeks, "test_fit", final_start, test_start,
            final_info, qtrain_final, qtest,
        ))
        model, _ = try_fit_hour_pooled_qcp(hourly, final_start, test_start)
        if model is None:
            raise ValueError("Direct QCP diagnostic fit unavailable")
        final_qcp_models[weeks] = (model, final_start)
        if weeks == 4:
            freshness, freshness_summary = _issue_freshness(
                hourly, model, pd.DatetimeIndex(rows.origin[test_mask].unique())
            )
        print(f"Prepared direct QCP {weeks}w final fit", flush=True)

        for method in DIRECT_METHODS:
            cv_maes: list[float] = []
            for candidate, leaf in enumerate(LEAVES, 1):
                fold_maes: list[float] = []
                for offset in CV_DAY_OFFSETS:
                    part = fold_cache[offset]
                    fit, valid = part["fit"], part["valid"]
                    xfit, xvalid = _direct_matrices(
                        method, fit, valid, part["qtrain"], part["qvalid"], weather_x
                    )
                    model = make_candidate("rf_history", {"min_samples_leaf": leaf})
                    model.fit(xfit, observed_labels[fit])
                    forecast = _candidate_forecast(model, xvalid)
                    mae = float(np.mean(np.abs(forecast - rows.y[valid])))
                    fold_maes.append(mae)
                    cv_records.append({
                        "train_weeks": weeks, "method": method,
                        "candidate": candidate, "fold_start": part["fold_start"],
                        "fold_end": part["fold_end"], "fit_start": part["train_start"],
                        "fit_end_exclusive": part["fold_start"],
                        "fit_pairs": int(fit.sum()),
                        "validation_pairs": int(valid.sum()),
                        "mae_kw": mae,
                        "params": json.dumps({"min_samples_leaf": leaf}, sort_keys=True),
                    })
                cv_maes.append(float(np.mean(fold_maes)))
                print(f"{weeks}w {method} leaf={leaf}: CV {cv_maes[-1]:.4f}", flush=True)
            best = int(np.argmin(cv_maes))
            leaf = LEAVES[best]
            selected[f"{weeks}w:{method}"] = {
                "params": {"min_samples_leaf": leaf},
                "cv_mae_kw": cv_maes[best], "cv_grid_mae_kw": cv_maes,
            }
            xfit, xtest = _direct_matrices(
                method, final_fit, test_mask, qtrain_final, qtest, weather_x
            )
            model = make_candidate("rf_history", {"min_samples_leaf": leaf})
            model.fit(xfit, observed_labels[final_fit])
            forecast = _candidate_forecast(model, xtest)
            predictions.append(_forecast_rows(rows, test_mask, forecast, method, weeks))
            print(f"{weeks}w {method}: test issued", flush=True)

    all_predictions = pd.concat(predictions, ignore_index=True)
    all_predictions = all_predictions.sort_values(
        ["train_weeks", "method", "origin", "lead"]
    )
    matched = all_predictions.loc[all_predictions["origin"].isin(common_origins)]
    for label, frame, count in (("full", all_predictions, 641),
                                ("matched", matched, 640)):
        groups = frame.groupby(["train_weeks", "method"])["origin"].nunique()
        if len(groups) != 3 * (len(REFERENCE_METHODS) + len(DIRECT_METHODS)) \
                or not groups.eq(count).all():
            raise ValueError(f"{label} direct-QCP predictions lack common issue coverage")
    threshold_kw = float(json.loads((core_dir / "metadata.json").read_text())[
        "peak_threshold_kw"
    ])
    full_scores, _ = score_predictions(all_predictions, threshold_kw)
    matched_scores, matched_lead = score_predictions(matched, threshold_kw)
    # The private OCHRE ER channel is first read here, after every candidate
    # model is fitted and every test forecast has been assembled and scored.
    diagnostic = pd.concat([
        _diagnose_completed_hours(
            hourly, test_start, test_end,
            source_model_start=final_qcp_models[weeks][1],
            model=final_qcp_models[weeks][0], train_weeks=weeks,
        )
        for weeks in TRAIN_WEEKS
    ], ignore_index=True)
    diagnostic_summary = _diagnostic_summary(diagnostic)
    if freshness is None or freshness_summary is None:
        raise ValueError("Four-week direct QCP freshness audit was not produced")
    output_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_dir / "predictions.csv.gz", "wt", encoding="utf-8") as stream:
        all_predictions.to_csv(stream, index=False)
    pd.DataFrame(cv_records).to_csv(output_dir / "rolling_cv.csv", index=False)
    pd.DataFrame(coverage_records).to_csv(output_dir / "qcp_fold_coverage.csv", index=False)
    full_scores.to_csv(output_dir / "full_horizon_scores.csv", index=False)
    matched_scores.to_csv(output_dir / "matched_horizon_scores.csv", index=False)
    matched_lead.to_csv(output_dir / "matched_lead_scores.csv", index=False)
    score_by_issue_day(matched, threshold_kw).to_csv(
        output_dir / "matched_issue_day_scores.csv", index=False
    )
    diagnostic.to_csv(output_dir / "qcp_completed_hour_diagnostic.csv", index=False)
    diagnostic_summary.to_csv(output_dir / "qcp_completed_hour_summary.csv", index=False)
    freshness.to_csv(output_dir / "qcp_test_issue_freshness.csv", index=False)
    (output_dir / "qcp_test_issue_freshness_summary.json").write_text(
        json.dumps(freshness_summary, indent=2) + "\n", encoding="utf-8"
    )
    metadata = {
        "target": target_audit["target"],
        "target_observation_mask": "originally observed Outside-Hub source bins only",
        "direct_qcp_source": "simulated total-home aggregate minus corrected Home Hub aggregate",
        "direct_qcp_feature": "approved symmetric signed heating estimate from latest complete ready source hour",
        "qcp_formulation": "existing 24-hour-group pooled QCP, fold fitted, prior-day walk-forward training features",
        "rf_spec": "80 trees, 0.8 max_features, 0.8 max_samples, bootstrap, seed 42",
        "rf_leaf_grid": LEAVES,
        "training_weeks": TRAIN_WEEKS,
        "cv_fold_day_offsets_in_week5": CV_DAY_OFFSETS,
        "full_test_origins": 641, "matched_test_origins": 640,
        "peak_threshold_kw": threshold_kw,
        "selected_validation_params": selected,
        "hidden_er_role": "offline QCP fidelity label only; never an RF/QCP input",
        "completed_hour_diagnostic": "signed QCP feature vs hidden ER hourly mean, week 6",
        "qcp_test_issue_freshness": freshness_summary,
        "raw_er_diagnostic_sha256": _sha256(BASELINE_DIR / "data/raw/home_01.csv"),
        **source_audit, **public_hashes,
        **baseline_provenance, **weather_provenance, **hybrid_provenance,
        "source_module_sha256": _sha256(Path(__file__)),
        "qcp_module_sha256": _sha256(Path(__file__).with_name("qcp_heating_feature.py")),
        "package_versions": {name: importlib.metadata.version(name)
                             for name in ("numpy", "pandas", "scikit-learn")},
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8"
    )
    _write_report(output_dir, matched_scores, selected,
                  baseline_provenance["saved_rf_selected_params"],
                  weather_provenance["saved_weather_rf_selected_params"],
                  pd.DataFrame(coverage_records), diagnostic_summary,
                  source_audit, freshness_summary)
    print(f"Saved direct-QCP RF comparison to {output_dir}", flush=True)


def _write_report(output_dir: Path, scores: pd.DataFrame,
                  selected: dict[str, dict[str, Any]],
                  base_selected: dict[str, dict[str, Any]],
                  weather_selected: dict[str, dict[str, Any]],
                  coverage: pd.DataFrame,
                  diagnostic: pd.DataFrame,
                  source_audit: dict[str, Any],
                  freshness: dict[str, Any]) -> None:
    eight = scores.loc[
        scores["scope"].eq("through_horizon") & scores["horizon_hours"].eq(8)
    ].set_index(["train_weeks", "method"])
    lines = [
        "# EV-only Outside-Hub plus resistance: direct aggregate QCP RF comparison", "",
        "The direct QCP sees the simulated **total-home minus corrected Home Hub** "
        "aggregate meter, not an HVAC resistance submeter. Input minutes retain "
        "only the saved source's approved observed/interpolated quality. The "
        "OCHRE plant already includes its exogenous Outside-Hub fill; no new "
        "aggregate interpolation is applied. For an interpolated source minute, "
        "availability is conservatively delayed until its closing originally "
        "observed endpoint. QCP fits and issued features use only four-bin hours "
        "whose power and weather have arrived by the respective cutoff.", "",
        "The signed QCP heating feature uses the approved 24 local-hour groups "
        "and symmetric observed-hour energy-minus-fitted-base rule below the "
        "heating change point. Negative estimates are retained. The RF never "
        "sees the hidden OCHRE resistance submeter; that channel is read only "
        "for the retrospective completed-hour diagnostic.", "",
        "All RFs retain the original observed-only target fit-label mask, "
        "2/3/4-week trailing windows, four rolling week-five validation day folds, "
        "and held-out week six. The two saved reference RFs are verified on the "
        "same 641 direct test issues. Headline scores restrict all four RFs to "
        "the 640 issues shared with the existing hybrid comparator.", "",
        "| Training | RF features | Week-5 CV MAE (kW) | Week-6 8h MAE (kW) | "
        "Bias (kW) | Underforecast | Peak recall | Peak shortfall (kW) |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for weeks in TRAIN_WEEKS:
        for method in (*REFERENCE_METHODS, *DIRECT_METHODS):
            row = eight.loc[(weeks, method)]
            choice = (
                base_selected[f"{weeks}w"] if method == REFERENCE_METHODS[0] else
                weather_selected[f"{weeks}w"] if method == REFERENCE_METHODS[1] else
                selected[f"{weeks}w:{method}"]
            )
            lines.append(
                f"| {weeks} weeks | {method} | {choice['cv_mae_kw']:.3f} | "
                f"{row['mae_kw']:.3f} | {row['bias_kw']:+.3f} | "
                f"{row['underforecast_rate']:.3f} | {row['peak_recall']:.3f} | "
                f"{row['actual_peak_mean_shortfall_kw']:.3f} |"
            )
    final = coverage.loc[coverage["phase"].eq("test_fit")]
    lines += [
        "", f"Approved aggregate bins: {source_audit['approved_15min_bins']}/4032; "
        f"complete QCP hours: {source_audit['complete_ready_qcp_hours']}/1008. "
        f"All final QCP fits had at least "
        f"{int(final['qcp_eligible_hour_groups'].min())}/24 eligible hour groups. "
        "Early training days without a QCP fit retain a zero-filled feature and "
        "an unavailable flag.", "",
        f"At the 641 week-six issue times, the four-week QCP source hour was "
        f"{freshness['median_age_minutes']:.0f} minutes old at the median, "
        f"{freshness['p95_age_minutes']:.0f} minutes at the 95th percentile, "
        f"and {freshness['max_age_minutes']:.0f} minutes at most; "
        f"{freshness['issues_older_than_one_hour']} issues used a source hour "
        "older than one hour. This is an audit of the existing no-freshness-cap "
        "issuer, not a new feature rule.", "",
        "## Completed-hour QCP diagnostic against hidden OCHRE resistance", "",
        "This check scores the signed feature for its own completed, ready "
        "source hour. It is retrospective feature fidelity, not an eight-hour "
        "forecast or a model input.", "",
        "| QCP fit window | Eligible test hours | MAE (kW) | Bias (kW) | "
        "Correlation | ER-on hours | On recall | Delayed hours |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in diagnostic.itertuples():
        lines.append(
            f"| {row.train_weeks} weeks | {row.eligible_test_hours}/168 | "
            f"{row.mae_kw:.3f} | {row.bias_kw:+.3f} | {row.correlation:.3f} | "
            f"{row.actual_er_on_hours} | {row.on_recall:.3f} | "
            f"{row.delayed_ready_hours} |"
        )
    lines += [
        "", "An actual peak is a 15-minute target mean at or above the fixed "
        "weeks-1–4 originally observed p95 threshold. Only seven distinct "
        "week-six bins cross it; overlapping forecast issues repeat those bins.", "",
        "[Matched scores](matched_horizon_scores.csv), "
        "[lead scores](matched_lead_scores.csv), "
        "[all 641 issues](full_horizon_scores.csv), "
        "[forecasts](predictions.csv.gz), "
        "[validation folds](rolling_cv.csv), "
        "[QCP coverage](qcp_fold_coverage.csv), "
        "[hourly diagnostic](qcp_completed_hour_diagnostic.csv), "
        "[issue freshness](qcp_test_issue_freshness.csv), and "
        "[provenance](metadata.json).", "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-dir", type=Path, default=CORE_DIR)
    parser.add_argument("--corrected-baseline-dir", type=Path,
                        default=CORRECTED_BASELINE_DIR)
    parser.add_argument("--hub-qcp-dir", type=Path, default=HUB_QCP_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
