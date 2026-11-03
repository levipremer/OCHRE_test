"""Compare causal, 12-hour DC House rest-of-home power forecasts.

Run directly for the August/September validation split. The separate
``run_late_september_test.py`` entry point keeps the later holdout distinct.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.ensemble import RandomForestRegressor

if __package__:
    from .feature_augmented_forest import fit_augmented_forest, predict_augmented_forest
    from .arx_one_lag import fit_arx_one_lag, predict_arx_one_lag
    from .arima_forecast import fit_arima
else:
    from feature_augmented_forest import fit_augmented_forest, predict_augmented_forest
    from arx_one_lag import fit_arx_one_lag, predict_arx_one_lag
    from arima_forecast import fit_arima


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "external_data" / "dc_house_power" / "rest_of_home_load.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "rest_of_home_forecasting" / "validation"
INTERVAL = pd.Timedelta(minutes=15)
SHORT_STEPS = 8
STEPS = 48
SCORE_HORIZONS = {"2h": SHORT_STEPS, "12h": STEPS}
TRAIN_DAYS = 20
CALIBRATION_DAYS = 6
CALIBRATION_TRAIN_DAYS = TRAIN_DAYS - CALIBRATION_DAYS
ORDINARY_QUANTILE = 0.75
FOREST_TREES = 100
FOREST_SEED = 42
METHODS = (
    "p95_14d", "p95_20d", "mean_14d", "mean_20d",
    "persistence", "random_forest", "rf_q75_peak", "rf_augmented",
    "arx_1", "arima_101", "arima_202",
)
METHOD_COLORS = {
    "p95_14d": "#2878B5",
    "p95_20d": "#E6862A",
    "mean_14d": "#17A0A4",
    "mean_20d": "#D45D79",
    "persistence": "#2A9D55",
    "random_forest": "#8A4AAF",
    "rf_q75_peak": "#B05B21",
    "rf_augmented": "#3E3E3E",
    "arx_1": "#176B76",
    "arima_101": "#894D69",
    "arima_202": "#BA4386",
    "arx_aic": "#6C7A28",
}
METHOD_LINESTYLES = {
    method: (
        "--" if method.startswith("mean_") else
        "-." if method == "rf_augmented" else
        ":" if method == "arx_1" else
        (0, (5, 2, 1, 2)) if method == "arima_101" else
        (0, (3, 2, 1, 2)) if method == "arima_202" else
        "-"
    )
    for method in METHODS
}
METHOD_LINESTYLES["arx_aic"] = (0, (6, 2))
PEAK_REFERENCE_START = pd.Timestamp("2026-08-02")
PEAK_REFERENCE_END = pd.Timestamp("2026-08-22")
MAE_PLOT_LIMIT_W = 6_000
TRANSPORT_PLOT_LIMIT_MIN = 720


def read_15_minute_power(path: Path) -> tuple[pd.Series, pd.Series, dict]:
    """Return complete approved bins and complete originally observed bins.

Source minute timestamps are treated as interval ends. A bin labelled 00:00
    averages the readings at 00:01 through 00:15 and becomes known at 00:15.
    """
    frame = pd.read_csv(path, parse_dates=["Time"])
    required = {"Time", "Value", "Quality"}
    if not required.issubset(frame):
        raise ValueError(f"Expected CSV columns {sorted(required)}")
    times = pd.DatetimeIndex(frame["Time"])
    if times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError("Minute timestamps must be unique and increasing")
    if not times.equals(pd.date_range(times[0], times[-1], freq="min")):
        raise ValueError("Source must retain the regular one-minute grid")
    values = pd.to_numeric(frame["Value"], errors="raise")
    if ((values.notna()) & (~np.isfinite(values) | values.lt(0))).any():
        raise ValueError("Power must be finite and nonnegative where present")
    quality = frame["Quality"].astype(str)
    valid_minute = quality.isin(("observed", "interpolated")) & values.notna()
    observed_minute = quality.eq("observed") & values.notna()
    if (quality.str.startswith("excluded") & values.notna()).any():
        raise ValueError("Excluded minute has a nonmissing power value")

    minutes = pd.DataFrame(
        {
            "approved": values.where(valid_minute).to_numpy(),
            "observed": values.where(observed_minute).to_numpy(),
        },
        index=times,
    )
    groups = minutes.resample("15min", closed="right", label="right")
    count = groups.size()
    approved = groups["approved"].mean().where(
        count.eq(15) & groups["approved"].count().eq(15)
    )
    observed = groups["observed"].mean().where(
        count.eq(15) & groups["observed"].count().eq(15)
    )
    approved.index = approved.index - INTERVAL
    observed.index = observed.index - INTERVAL
    if not approved.index.equals(observed.index):
        raise AssertionError("Approved and observed bins are misaligned")
    details = {
        "source_minutes": len(frame),
        "approved_15min_bins": int(approved.notna().sum()),
        "observed_15min_bins": int(observed.notna().sum()),
        "quality_counts": {str(k): int(v) for k, v in quality.value_counts().items()},
    }
    return approved, observed, details


def features_from_observed(observed: pd.Series) -> pd.DataFrame:
    """Features for bin start t use only bins strictly before t."""
    previous = observed.shift(1)
    recent_30m = previous.rolling(2, min_periods=1)
    recent_1h = previous.rolling(4, min_periods=1)
    recent_2h = previous.rolling(8, min_periods=1)
    clock_angle = 2 * np.pi * (
        observed.index.hour * 60 + observed.index.minute
    ) / 1440
    week_angle = 2 * np.pi * observed.index.dayofweek / 7
    return pd.DataFrame(
        {
            "latest_w": previous,
            "mean_30m_w": recent_30m.mean(),
            "mean_1h_w": recent_1h.mean(),
            "mean_2h_w": recent_2h.mean(),
            "max_1h_w": recent_1h.max(),
            "max_2h_w": recent_2h.max(),
            "min_1h_w": recent_1h.min(),
            "observed_count_1h": recent_1h.count(),
            "observed_count_2h": recent_2h.count(),
            "clock_sin": np.sin(clock_angle),
            "clock_cos": np.cos(clock_angle),
            "week_sin": np.sin(week_angle),
            "week_cos": np.cos(week_angle),
        },
        index=observed.index,
    )


def fit_forest(observed: pd.Series, train_end: pd.Timestamp,
               train_days: int = TRAIN_DAYS) -> tuple[RandomForestRegressor, dict]:
    train_start = train_end - pd.Timedelta(days=train_days)
    window = observed.loc[(observed.index >= train_start) & (observed.index < train_end)]
    if len(window) != train_days * 96:
        raise ValueError(
            f"Random forest requires exactly {train_days} calendar days of 15-minute slots"
        )
    features = features_from_observed(window)
    available_features = features["latest_w"].notna()
    x_parts: list[pd.DataFrame] = []
    y_parts: list[pd.Series] = []
    counts_by_lead: dict[str, int] = {}
    for lead in range(1, STEPS + 1):
        target = window.shift(-(lead - 1))
        usable = available_features & target.notna()
        x_part = features.loc[usable].copy()
        x_part["lead_step"] = lead
        x_parts.append(x_part)
        y_parts.append(target.loc[usable])
        counts_by_lead[str(lead)] = int(usable.sum())
    x = pd.concat(x_parts, ignore_index=True)
    y = pd.concat(y_parts, ignore_index=True)
    if len(x) < 1000 or not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError("Insufficient complete observed training examples")
    model = RandomForestRegressor(
        n_estimators=FOREST_TREES,
        min_samples_leaf=3,
        random_state=FOREST_SEED,
        n_jobs=int(os.environ.get("HEMS_RF_THREADS", "-1")),
    )
    model.fit(x, y)
    return model, {
        "start_inclusive": train_start.isoformat(),
        "end_exclusive": train_end.isoformat(),
        "calendar_days": train_days,
        "observed_bins": int(window.notna().sum()),
        "training_rows": len(x),
        "training_rows_by_lead": counts_by_lead,
        "feature_columns": list(x.columns),
        "trees": FOREST_TREES,
        "min_samples_leaf": 3,
        "random_seed": FOREST_SEED,
        "fit_schedule": "once before the evaluation period",
    }


def fit_nonpeak_residual_quantiles(
    observed: pd.Series, train_end: pd.Timestamp, threshold_w: float
) -> tuple[np.ndarray, dict]:
    """Calibrate each RF lead on later, originally observed nonpeak targets.

    A forest fit to the first 14 days predicts targets in the final six days
    of the 20-day training period. The final evaluation forest is fit
    separately on all 20 days. Each calibration target and the latest feature
    bin must be originally observed and below the peak threshold, so active
    peaks and peak-to-nonpeak transitions cannot set the ordinary-load offset.
    """
    calibration_start = train_end - pd.Timedelta(days=CALIBRATION_DAYS)
    calibration_model, calibration_fit = fit_forest(
        observed, calibration_start, train_days=CALIBRATION_TRAIN_DAYS
    )
    train_start = train_end - pd.Timedelta(days=TRAIN_DAYS)
    window = observed.loc[(observed.index >= train_start) & (observed.index < train_end)]
    features = features_from_observed(window)
    origins = features.loc[
        (features.index >= calibration_start) & (features.index < train_end)
    ]
    origins = origins.loc[
        origins["latest_w"].notna() & (origins["latest_w"] < threshold_w)
    ]
    if origins.empty or not np.isfinite(origins.to_numpy(dtype=float)).all():
        raise ValueError("No complete, observed nonpeak calibration origins")
    feature_parts: list[pd.DataFrame] = []
    target_parts: list[np.ndarray] = []
    lead_parts: list[np.ndarray] = []
    for lead in range(1, STEPS + 1):
        target_index = origins.index + (lead - 1) * INTERVAL
        targets = window.reindex(target_index).to_numpy(dtype=float)
        usable = np.isfinite(targets) & (targets < threshold_w)
        if not usable.any():
            raise ValueError(f"No originally observed nonpeak targets for lead {lead}")
        feature_part = origins.iloc[np.flatnonzero(usable)].copy()
        feature_part["lead_step"] = lead
        feature_parts.append(feature_part)
        target_parts.append(targets[usable])
        lead_parts.append(np.repeat(lead, int(usable.sum())))
    calibration_features = pd.concat(feature_parts, ignore_index=True)
    calibration_targets = np.concatenate(target_parts)
    calibration_leads = np.concatenate(lead_parts)
    predictions = calibration_model.predict(calibration_features)
    residuals = calibration_targets - predictions
    raw_offsets = np.array([
        np.quantile(residuals[calibration_leads == lead], ORDINARY_QUANTILE)
        for lead in range(1, STEPS + 1)
    ])
    offsets = np.maximum(raw_offsets, 0.0)
    return offsets, {
        "rule": "max(0, lead-wise 75th percentile of observed nonpeak target minus chronological RF prediction)",
        "calibration_fit": calibration_fit,
        "calibration_start_inclusive": calibration_start.isoformat(),
        "calibration_end_exclusive": train_end.isoformat(),
        "calibration_days": CALIBRATION_DAYS,
        "calibration_origins_with_observed_nonpeak_latest": len(origins),
        "calibration_nonpeak_pairs": len(calibration_targets),
        "calibration_nonpeak_pairs_by_lead": {
            str(lead): int((calibration_leads == lead).sum())
            for lead in range(1, STEPS + 1)
        },
        "raw_residual_quantile_w_by_lead": raw_offsets.tolist(),
        "applied_nonnegative_offset_w_by_lead": offsets.tolist(),
    }


def observed_peak_run_age(
    observed: pd.Series, origin: pd.Timestamp, threshold_w: float
) -> tuple[int, bool]:
    """Age of the peak at issue, using consecutive observed bins only.

    The Boolean says whether an observed nonpeak establishes the run's start.
    A gap before the run makes the age a lower bound, rather than evidence of
    a newly started episode.
    """
    latest_position = observed.index.get_loc(origin - INTERVAL)
    values = observed.to_numpy(dtype=float)
    position = int(latest_position)
    age = 0
    while position >= 0 and np.isfinite(values[position]) and values[position] >= threshold_w:
        age += 1
        position -= 1
    known_onset = (
        position >= 0 and np.isfinite(values[position]) and values[position] < threshold_w
    )
    return age, bool(known_onset) if age else False


def fit_peak_continuation(
    observed: pd.Series, train_end: pd.Timestamp, threshold_w: float
) -> tuple[dict[str, np.ndarray], dict]:
    """Estimate empirical same-episode continuation, censoring missing ends.

    Each observed peak bin is an issue state. A future observed nonpeak ends
    that episode and establishes zeros thereafter. A missing bin before an
    observed end censors that issue state at that lead, so the gap never
    masquerades as an ended peak. Small known-age groups borrow the pooled
    curve lead by lead. The cumulative minimum keeps each continuation curve
    nonincreasing despite changes in the classifiable denominator.
    """
    train_start = train_end - pd.Timedelta(days=TRAIN_DAYS)
    window = observed.loc[(observed.index >= train_start) & (observed.index < train_end)]
    if len(window) != TRAIN_DAYS * 96:
        raise ValueError("Peak continuation needs the full 20-day calendar grid")
    values = window.to_numpy(dtype=float)
    peak_indices = np.flatnonzero(np.isfinite(values) & (values >= threshold_w))
    if not len(peak_indices):
        raise ValueError("No originally observed training peaks for continuation")
    samples: dict[str, list[np.ndarray]] = {"pooled": [], "age_1": [], "age_2plus": []}
    unknown_onset_origins = 0
    for position in peak_indices:
        onset_index = position - 1
        while onset_index >= 0 and np.isfinite(values[onset_index]) and values[onset_index] >= threshold_w:
            onset_index -= 1
        age = position - onset_index
        known_onset = (
            onset_index >= 0 and np.isfinite(values[onset_index])
            and values[onset_index] < threshold_w
        )
        if not known_onset:
            unknown_onset_origins += 1
        outcomes = np.full(STEPS, np.nan)
        ended = False
        for lead in range(1, STEPS + 1):
            target_position = position + lead
            if ended:
                outcomes[lead - 1] = 0.0
            elif target_position >= len(values) or not np.isfinite(values[target_position]):
                break
            elif values[target_position] < threshold_w:
                outcomes[lead - 1] = 0.0
                ended = True
            else:
                outcomes[lead - 1] = 1.0
        samples["pooled"].append(outcomes)
        if known_onset:
            samples["age_1" if age == 1 else "age_2plus"].append(outcomes)

    matrices = {
        key: np.stack(rows) if rows else np.empty((0, STEPS))
        for key, rows in samples.items()
    }
    counts = {key: np.isfinite(matrix).sum(axis=0) for key, matrix in matrices.items()}
    probabilities = {
        key: np.divide(
            np.nansum(matrix, axis=0), counts[key],
            out=np.full(STEPS, np.nan), where=counts[key] > 0,
        )
        for key, matrix in matrices.items()
    }
    min_group_count = 5
    curves: dict[str, np.ndarray] = {}
    fallback_by_group: dict[str, int] = {}
    for key in samples:
        sufficient = counts[key] >= min_group_count if key != "pooled" else counts[key] > 0
        leadwise = np.where(sufficient, probabilities[key], probabilities["pooled"])
        leadwise = np.nan_to_num(leadwise, nan=0.0)
        curves[key] = np.minimum.accumulate(leadwise)
        fallback_by_group[key] = int((~sufficient).sum()) if key != "pooled" else 0
    return curves, {
        "rule": "fraction of originally observed peak origins still in the same episode at each lead among origins with known status at that lead; an observed end stays ended, missing before end right-censors",
        "train_start_inclusive": train_start.isoformat(),
        "train_end_exclusive": train_end.isoformat(),
        "training_peak_origins": len(peak_indices),
        "unknown_onset_peak_origins": unknown_onset_origins,
        "known_age_1_origins": len(samples["age_1"]),
        "known_age_2plus_origins": len(samples["age_2plus"]),
        "minimum_age_group_classifiable_origins_per_lead": min_group_count,
        "age_group_fallback_leads": fallback_by_group,
        "classifiable_origins_by_lead": {
            key: counts[key].astype(int).tolist() for key in samples
        },
        "raw_continuation_probability_by_lead": {
            key: [float(value) if np.isfinite(value) else None for value in probabilities[key]]
            for key in samples
        },
        "applied_continuation_probability_by_lead": {
            key: curves[key].tolist() for key in samples
        },
    }


def conservative_peak_forecast(
    rf_forecast_w: np.ndarray, q75_residual_w: np.ndarray,
    latest_w: float, threshold_w: float, continuation_probability: np.ndarray
) -> np.ndarray:
    """RF ordinary trend plus a nonnegative active-peak continuation term."""
    baseline = np.maximum(np.asarray(rf_forecast_w) + np.asarray(q75_residual_w), 0.0)
    if latest_w < threshold_w:
        return baseline
    current_peak_excess = max(latest_w - baseline[0], 0.0)
    return baseline + current_peak_excess * np.asarray(continuation_probability)


def time_of_day_p95(approved: pd.Series, observed: pd.Series,
                    origin: pd.Timestamp, days: int) -> pd.Series:
    """Slotwise 95th percentile from completed calendar days before origin."""
    cutoff = origin.normalize()
    start = cutoff - pd.Timedelta(days=days)
    history = approved.loc[(approved.index >= start) & (approved.index < cutoff)].copy()
    if len(history) != days * 96:
        raise ValueError(f"The {days}-day percentile history is incomplete at {origin}")
    # A filled bin immediately before midnight might rely on a gap-closing
    # meter reading after this midnight issue time. Keep only originally
    # observed bins in that final hour; older approved fills are available.
    recent = history.index >= cutoff - pd.Timedelta(hours=1)
    history.loc[recent & observed.reindex(history.index).isna()] = np.nan
    slot = history.index.hour * 4 + history.index.minute // 15
    profile = history.groupby(slot).quantile(0.95, interpolation="linear")
    if len(profile) != 96 or profile.isna().any():
        raise ValueError(f"The {days}-day percentile history lacks a clock slot at {origin}")
    return profile


def time_of_day_mean(approved: pd.Series, observed: pd.Series,
                     origin: pd.Timestamp, days: int) -> pd.Series:
    """Slotwise arithmetic mean from completed calendar days before origin."""
    cutoff = origin.normalize()
    start = cutoff - pd.Timedelta(days=days)
    history = approved.loc[(approved.index >= start) & (approved.index < cutoff)].copy()
    if len(history) != days * 96:
        raise ValueError(f"The {days}-day mean history is incomplete at {origin}")
    # Match the percentile method's safeguard against fills whose closing
    # reading could arrive after the daily midnight issue time.
    recent = history.index >= cutoff - pd.Timedelta(hours=1)
    history.loc[recent & observed.reindex(history.index).isna()] = np.nan
    slot = history.index.hour * 4 + history.index.minute // 15
    profile = history.groupby(slot).mean()
    if len(profile) != 96 or profile.isna().any():
        raise ValueError(f"The {days}-day mean history lacks a clock slot at {origin}")
    return profile


def remaining_peak_steps(values: np.ndarray, threshold_w: float) -> int:
    above = np.asarray(values) >= threshold_w
    first_end = np.flatnonzero(~above)
    return int(first_end[0]) if len(first_end) else len(above)


def actual_remaining_peak_steps(values: np.ndarray, threshold_w: float) -> tuple[int | None, bool]:
    """Return the observed run through its end or the supplied horizon cap.

    A missing bin after an observed peak end does not obscure that end.
    Missingness before the end makes the duration unknown.
    """
    for step, value in enumerate(values):
        if not np.isfinite(value):
            return None, False
        if value < threshold_w:
            return step, False
    return len(values), True


def safe_mean(values: pd.Series) -> float:
    return float(values.mean()) if len(values) else float("nan")


def score_predictions(predictions: pd.DataFrame, threshold_w: float,
                      methods: tuple[str, ...] = METHODS) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored = predictions.loc[predictions["actual_w"].notna()].copy()
    if scored.empty:
        raise ValueError("No observed or approved-interpolation target bins were available to score")
    scored["error_w"] = scored["forecast_w"] - scored["actual_w"]
    scored["actual_peak"] = scored["actual_w"] >= threshold_w
    scored["forecast_peak"] = scored["forecast_w"] >= threshold_w

    def measures(part: pd.DataFrame) -> dict:
        peak = part.loc[part["actual_peak"]]
        ongoing_peak = part.loc[part["origin_peak"]]
        predicted_peak = part.loc[part["forecast_peak"]]
        err = part["error_w"]
        return {
            "scored_predictions": len(part),
            "actual_peak_predictions": len(peak),
            "ongoing_peak_predictions": len(ongoing_peak),
            "forecast_peak_predictions": len(predicted_peak),
            "mae_w": safe_mean(err.abs()),
            "rmse_w": float(np.sqrt((err**2).mean())) if len(err) else float("nan"),
            "bias_w": safe_mean(err),
            "peak_mae_w": safe_mean(peak["error_w"].abs()),
            "peak_rmse_w": float(np.sqrt((peak["error_w"]**2).mean())) if len(peak) else float("nan"),
            "ongoing_peak_mae_w": safe_mean(ongoing_peak["error_w"].abs()),
            "ongoing_peak_rmse_w": float(np.sqrt((ongoing_peak["error_w"]**2).mean())) if len(ongoing_peak) else float("nan"),
            "peak_recall": safe_mean(peak["forecast_peak"].astype(float)),
            "peak_precision": safe_mean(predicted_peak["actual_peak"].astype(float)),
        }

    overall = pd.DataFrame(
        [{"method": method, **measures(scored.loc[scored["method"] == method])} for method in methods]
    )
    by_lead = pd.DataFrame(
        [
            {"method": method, "lead_minutes": lead * 15,
             **measures(scored.loc[(scored["method"] == method) & (scored["lead_minutes"] == lead * 15)])}
            for method in methods for lead in range(1, STEPS + 1)
        ]
    )
    return overall, by_lead


def temporal_wasserstein_minutes(forecast_energy_wh: np.ndarray,
                                 actual_energy_wh: np.ndarray) -> float:
    """Move energy between 15-minute time locations; return normalized work in minutes.

    Each nonempty profile is normalized to unit mass by scipy. Consequently this
    compares timing and shape, while separate scores retain energy and peak height.
    A profile with zero mass has no defined transport distance.
    """
    forecast = np.asarray(forecast_energy_wh, dtype=float)
    actual = np.asarray(actual_energy_wh, dtype=float)
    if forecast.ndim != 1 or actual.shape != forecast.shape or not len(forecast):
        raise ValueError("Transport profiles must have the same nonzero number of time bins")
    if not (np.isfinite(forecast).all() and np.isfinite(actual).all()):
        raise ValueError("Transport profiles must be complete and finite")
    if (forecast < 0).any() or (actual < 0).any():
        raise ValueError("Transport masses must be nonnegative")
    if forecast.sum() <= 0 or actual.sum() <= 0:
        return float("nan")
    times_min = np.arange(len(forecast), dtype=float) * (INTERVAL.total_seconds() / 60)
    return float(wasserstein_distance(
        times_min, times_min, u_weights=forecast, v_weights=actual
    ))


def transport_window_row(forecast_w: np.ndarray, actual_w: np.ndarray,
                         threshold_w: float) -> dict:
    """Score two complete, aligned power profiles over one forecast window."""
    forecast = np.asarray(forecast_w, dtype=float)
    actual = np.asarray(actual_w, dtype=float)
    interval_hours = INTERVAL.total_seconds() / 3600
    forecast_excess_wh = np.maximum(forecast - threshold_w, 0.0) * interval_hours
    actual_excess_wh = np.maximum(actual - threshold_w, 0.0) * interval_hours
    forecast_peak_energy_wh = float(forecast_excess_wh.sum())
    actual_peak_energy_wh = float(actual_excess_wh.sum())
    energy_error_wh = forecast_peak_energy_wh - actual_peak_energy_wh
    if forecast_peak_energy_wh > 0 and actual_peak_energy_wh > 0:
        peak_mass_case = "both"
    elif actual_peak_energy_wh > 0:
        peak_mass_case = "actual_only"
    elif forecast_peak_energy_wh > 0:
        peak_mass_case = "forecast_only"
    else:
        peak_mass_case = "neither"
    return {
        "raw_w1_min": temporal_wasserstein_minutes(
            forecast * interval_hours, actual * interval_hours
        ),
        "peak_excess_w1_min": temporal_wasserstein_minutes(
            forecast_excess_wh, actual_excess_wh
        ),
        "peak_mass_case": peak_mass_case,
        "actual_peak_window": bool((actual >= threshold_w).any()),
        "forecast_peak_window": bool((forecast >= threshold_w).any()),
        "actual_peak_power_w": float(actual.max()),
        "forecast_peak_power_w": float(forecast.max()),
        "peak_height_error_w": float(forecast.max() - actual.max()),
        "actual_peak_excess_energy_wh": actual_peak_energy_wh,
        "forecast_peak_excess_energy_wh": forecast_peak_energy_wh,
        "peak_excess_energy_error_wh": energy_error_wh,
        "missing_peak_energy_wh": max(-energy_error_wh, 0.0),
        "extra_peak_energy_wh": max(energy_error_wh, 0.0),
    }


def summarize_transport(windows: pd.DataFrame,
                        methods: tuple[str, ...] = METHODS) -> pd.DataFrame:
    """Keep zero-mass and incomplete-window counts visible beside defined W1s."""
    rows: list[dict] = []
    for horizon in SCORE_HORIZONS:
        horizon_rows = windows.loc[windows["horizon"] == horizon]
        for cohort in ("all", "actual_peak", "origin_peak"):
            if cohort == "actual_peak":
                cohort_rows = horizon_rows.loc[horizon_rows["actual_peak_window"]]
            elif cohort == "origin_peak":
                cohort_rows = horizon_rows.loc[horizon_rows["origin_peak"]]
            else:
                cohort_rows = horizon_rows
            for method in methods:
                part = cohort_rows.loc[cohort_rows["method"] == method]
                case_counts = part["peak_mass_case"].value_counts()
                rows.append({
                    "horizon": horizon,
                    "cohort": cohort,
                    "method": method,
                    "complete_windows": len(part),
                    "raw_w1_defined_windows": int(part["raw_w1_min"].notna().sum()),
                    "raw_w1_mean_min": safe_mean(part["raw_w1_min"]),
                    "peak_w1_defined_windows": int(part["peak_excess_w1_min"].notna().sum()),
                    "peak_excess_w1_mean_min": safe_mean(part["peak_excess_w1_min"]),
                    "both_peak_mass_windows": int(case_counts.get("both", 0)),
                    "actual_only_peak_mass_windows": int(case_counts.get("actual_only", 0)),
                    "forecast_only_peak_mass_windows": int(case_counts.get("forecast_only", 0)),
                    "neither_peak_mass_windows": int(case_counts.get("neither", 0)),
                    "peak_height_mae_w": safe_mean(part["peak_height_error_w"].abs()),
                    "peak_height_bias_w": safe_mean(part["peak_height_error_w"]),
                    "mean_missing_peak_energy_wh": safe_mean(part["missing_peak_energy_wh"]),
                    "mean_extra_peak_energy_wh": safe_mean(part["extra_peak_energy_wh"]),
                    "mean_absolute_peak_energy_error_wh": safe_mean(
                        part["peak_excess_energy_error_wh"].abs()
                    ),
                })
    return pd.DataFrame(rows)


def make_plot(by_lead: pd.DataFrame, output: Path, split: str,
              horizon: str, y_limit_w: float,
              methods: tuple[str, ...] = METHODS) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharex=True, sharey=True, constrained_layout=True)
    for method in methods:
        group = by_lead.loc[by_lead["method"] == method]
        style = {"marker": "o", "label": method, "color": METHOD_COLORS[method],
                 "linestyle": METHOD_LINESTYLES[method]}
        axes[0].plot(group["lead_minutes"], group["mae_w"], **style)
        axes[1].plot(group["lead_minutes"], group["peak_mae_w"], **style)
        axes[2].plot(group["lead_minutes"], group["ongoing_peak_mae_w"], **style)
    axes[0].set_title("All scored targets")
    axes[1].set_title("Actual target is a peak")
    axes[2].set_title("Peak ongoing at forecast origin")
    for ax in axes:
        ax.set_xlabel("Forecast lead (minutes)")
        if horizon == "2h":
            ax.set_xticks(range(15, 121, 15))
        else:
            ax.set_xticks(range(60, 721, 60))
        ax.set_xlim(15, SCORE_HORIZONS[horizon] * 15)
        ax.set_ylim(0, y_limit_w)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Mean absolute error (W)")
    axes[0].legend(fontsize=8, loc="upper right")
    figure.suptitle(
        f"Rest-of-home forecast errors: {split.replace('_', ' ')}, {horizon} window"
    )
    suffix = "" if horizon == "2h" else f"_{horizon}"
    figure.savefig(output / f"{split}_mae_by_lead{suffix}.png", dpi=160)
    plt.close(figure)


def make_transport_plot(summary: pd.DataFrame, output: Path, split: str,
                        horizon: str, y_limit_min: float,
                        methods: tuple[str, ...] = METHODS) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    figure, axes = plt.subplots(2, 3, figsize=(18, 8), sharey=True, constrained_layout=True)
    cohort_titles = {
        "all": "All complete windows",
        "actual_peak": "Realized peak in window",
        "origin_peak": "Peak ongoing at issue",
    }
    for col, cohort in enumerate(cohort_titles):
        part = summary.loc[(summary["horizon"] == horizon) & (summary["cohort"] == cohort)]
        part = part.set_index("method").reindex(methods)
        for row, (field, label) in enumerate((
            ("raw_w1_mean_min", "Full-load W1"),
            ("peak_excess_w1_mean_min", "Peak-excess W1"),
        )):
            ax = axes[row, col]
            values = part[field].to_numpy(dtype=float)
            counts = part[
                "raw_w1_defined_windows" if row == 0 else "peak_w1_defined_windows"
            ].to_numpy(dtype=int)
            x = np.arange(len(methods))
            ax.bar(x, np.nan_to_num(values, nan=0.0),
                   color=[METHOD_COLORS[method] for method in methods], width=0.72)
            for pos, value, count in zip(x, values, counts):
                caption = f"n={count}" if np.isfinite(value) else "N/A"
                label_y = (min(y_limit_min * 0.98, max(value + 1, value * 1.12))
                           if np.isfinite(value) else 0.8)
                ax.text(pos, label_y,
                        caption, ha="center", va="bottom", fontsize=7, rotation=90)
            ax.set_xticks(x, methods, rotation=35, ha="right", fontsize=8)
            ax.set_yscale("symlog", linthresh=10)
            ax.set_ylim(0, y_limit_min)
            ax.set_yticks([0, 2, 5, 10, 30, 100, 300, 720])
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            ax.grid(axis="y", alpha=0.2)
            if row == 0:
                ax.set_title(cohort_titles[cohort])
            if col == 0:
                ax.set_ylabel(f"{label} (minutes; symlog axis)")
    figure.suptitle(
        f"Time-aware Wasserstein distance: {split}, {horizon} complete windows\n"
        "Bars average only defined distances; n gives the number of windows; common nonlinear minute axis"
    )
    figure.savefig(output / f"{split}_wasserstein_{horizon}.png", dpi=160)
    plt.close(figure)


def run_evaluation(*, data: Path, output: Path, start: pd.Timestamp,
                   end: pd.Timestamp, split: str) -> dict:
    if start != start.normalize() or end != end.normalize() or end <= start:
        raise ValueError("Evaluation start/end must be increasing midnight timestamps")
    approved, observed, source_details = read_15_minute_power(data)
    train_start = start - pd.Timedelta(days=TRAIN_DAYS)
    if observed.index[0] > train_start or approved.index[-1] < end - INTERVAL:
        raise ValueError("Source lacks the requested training or evaluation period")
    reference = observed.loc[
        (observed.index >= PEAK_REFERENCE_START) & (observed.index < PEAK_REFERENCE_END)
    ].dropna()
    if len(reference) < 100:
        raise ValueError("Insufficient observed pre-validation data for the peak threshold")
    threshold_w = float(reference.quantile(0.95, interpolation="linear"))
    q75_offsets, q75_details = fit_nonpeak_residual_quantiles(observed, start, threshold_w)
    continuation_curves, continuation_details = fit_peak_continuation(
        observed, start, threshold_w
    )
    model, model_details = fit_forest(observed, start)
    augmented_model, augmented_details = fit_augmented_forest(
        observed, start, threshold_w, train_days=TRAIN_DAYS, steps=STEPS
    )
    arx_model, arx_details = fit_arx_one_lag(observed, start, train_days=TRAIN_DAYS)
    arima_model, arima_details = fit_arima(observed, start, train_days=TRAIN_DAYS)
    arima_202_model, arima_202_details = fit_arima(
        observed, start, train_days=TRAIN_DAYS, order=(2, 0, 2), maxiter=200
    )
    feature_rows = features_from_observed(observed)
    # Keep every origin that has a full first-two-hour window inside the split.
    # Only earlier origins qualify for the separate full-12-hour scores.
    last_origin = end - SHORT_STEPS * INTERVAL
    last_long_origin = end - STEPS * INTERVAL
    origins = pd.date_range(start, last_origin, freq=INTERVAL)
    daily_profiles: dict[tuple[pd.Timestamp, int], pd.Series] = {}
    daily_means: dict[tuple[pd.Timestamp, int], pd.Series] = {}
    prediction_parts: list[pd.DataFrame] = []
    transport_rows: list[dict] = []
    duration_rows: dict[str, list[dict]] = {horizon: [] for horizon in SCORE_HORIZONS}
    eligible_origins = 0
    eligible_12h_origins = 0
    ongoing_peak_origins = 0
    complete_window_origins = {horizon: 0 for horizon in SCORE_HORIZONS}
    duration_incomplete_origins = {horizon: 0 for horizon in SCORE_HORIZONS}
    duration_censored_origins = {horizon: 0 for horizon in SCORE_HORIZONS}
    for origin in origins:
        feature = feature_rows.loc[origin]
        if not np.isfinite(feature.to_numpy(dtype=float)).all():
            continue
        eligible_origins += 1
        if origin <= last_long_origin:
            eligible_12h_origins += 1
        future = pd.date_range(origin, periods=STEPS, freq=INTERVAL)
        actual = approved.reindex(future).to_numpy(dtype=float)
        truth_observed = observed.reindex(future).notna().to_numpy()
        outside_split = future >= end
        actual[outside_split] = np.nan
        truth_observed[outside_split] = False
        latest = float(feature["latest_w"])
        forecasts: dict[str, np.ndarray] = {}
        slots = future.hour * 4 + future.minute // 15
        for days in (14, 20):
            key = (origin.normalize(), days)
            if key not in daily_profiles:
                daily_profiles[key] = time_of_day_p95(approved, observed, origin, days)
            forecasts[f"p95_{days}d"] = daily_profiles[key].reindex(slots).to_numpy(dtype=float)
            if key not in daily_means:
                daily_means[key] = time_of_day_mean(approved, observed, origin, days)
            forecasts[f"mean_{days}d"] = daily_means[key].reindex(slots).to_numpy(dtype=float)
        forecasts["persistence"] = np.repeat(latest, STEPS)
        rf_input = pd.DataFrame([feature.to_dict()] * STEPS)
        rf_input["lead_step"] = np.arange(1, STEPS + 1)
        forecasts["random_forest"] = np.maximum(model.predict(rf_input), 0.0)
        if latest >= threshold_w:
            run_age, onset_known = observed_peak_run_age(observed, origin, threshold_w)
            curve_key = ("age_1" if run_age == 1 else "age_2plus") if onset_known else "pooled"
        else:
            curve_key = "pooled"
        forecasts["rf_q75_peak"] = conservative_peak_forecast(
            forecasts["random_forest"], q75_offsets, latest, threshold_w,
            continuation_curves[curve_key],
        )
        forecasts["rf_augmented"] = predict_augmented_forest(
            augmented_model, observed, origin, threshold_w, steps=STEPS
        )
        forecasts["arx_1"] = predict_arx_one_lag(
            arx_model, observed, origin, steps=STEPS
        )
        forecasts["arima_101"] = arima_model.forecast(
            observed, origin, steps=STEPS
        )
        forecasts["arima_202"] = arima_202_model.forecast(
            observed, origin, steps=STEPS
        )
        for method in METHODS:
            prediction_parts.append(pd.DataFrame({
                "method": method,
                "origin": origin,
                "target_start": future,
                "lead_minutes": np.arange(1, STEPS + 1) * 15,
                "actual_w": actual,
                "forecast_w": forecasts[method],
                "truth_fully_observed": truth_observed,
                "origin_power_w": latest,
                "origin_peak": latest >= threshold_w,
                "full_12h_within_split": origin <= last_long_origin,
            }))

        if latest >= threshold_w:
            ongoing_peak_origins += 1
        for horizon, steps in SCORE_HORIZONS.items():
            if origin > end - steps * INTERVAL:
                continue
            horizon_actual = actual[:steps]
            if np.isfinite(horizon_actual).all():
                complete_window_origins[horizon] += 1
                for method in METHODS:
                    transport_rows.append({
                        "horizon": horizon,
                        "origin": origin,
                        "method": method,
                        "origin_peak": latest >= threshold_w,
                        **transport_window_row(
                            forecasts[method][:steps], horizon_actual, threshold_w
                        ),
                    })
            if latest < threshold_w:
                continue
            actual_steps, right_censored = actual_remaining_peak_steps(
                horizon_actual, threshold_w
            )
            if actual_steps is None:
                duration_incomplete_origins[horizon] += 1
                continue
            if right_censored:
                duration_censored_origins[horizon] += 1
            for method in METHODS:
                predicted_steps = remaining_peak_steps(forecasts[method][:steps], threshold_w)
                duration_rows[horizon].append({
                    "method": method,
                    "origin": origin,
                    "actual_remaining_minutes": actual_steps * 15,
                    "predicted_remaining_minutes": predicted_steps * 15,
                    "error_minutes": (predicted_steps - actual_steps) * 15,
                    "actual_right_censored_at_horizon": right_censored,
                })

    if not prediction_parts:
        raise ValueError("No forecast origins have an observed latest 15-minute bin")
    predictions = pd.concat(prediction_parts, ignore_index=True)
    predictions["scored"] = predictions["actual_w"].notna()
    predictions["actual_peak"] = predictions["actual_w"] >= threshold_w
    predictions["forecast_peak"] = predictions["forecast_w"] >= threshold_w
    metrics_by_horizon: dict[str, pd.DataFrame] = {}
    by_lead_by_horizon: dict[str, pd.DataFrame] = {}
    duration_by_horizon: dict[str, pd.DataFrame] = {}
    for horizon, steps in SCORE_HORIZONS.items():
        part = predictions.loc[predictions["lead_minutes"] <= steps * 15]
        if horizon == "12h":
            part = part.loc[part["full_12h_within_split"]]
        metrics, by_lead = score_predictions(part, threshold_w)
        duration = pd.DataFrame(duration_rows[horizon], columns=[
            "method", "origin", "actual_remaining_minutes", "predicted_remaining_minutes",
            "error_minutes", "actual_right_censored_at_horizon"
        ])
        for method in METHODS:
            group = duration.loc[duration["method"] == method]
            metrics.loc[metrics["method"] == method, "duration_scored_origins"] = len(group)
            metrics.loc[metrics["method"] == method, "remaining_duration_mae_min"] = safe_mean(
                group["error_minutes"].abs()
            )
            metrics.loc[metrics["method"] == method, "remaining_duration_bias_min"] = safe_mean(
                group["error_minutes"]
            )
            metrics.loc[metrics["method"] == method, "remaining_duration_within_15min"] = safe_mean(
                group["error_minutes"].abs().le(15).astype(float)
            )
            ended = group.loc[~group["actual_right_censored_at_horizon"]]
            metrics.loc[metrics["method"] == method, "ended_peak_duration_mae_min"] = safe_mean(
                ended["error_minutes"].abs()
            )
        metrics_by_horizon[horizon] = metrics
        by_lead_by_horizon[horizon] = by_lead
        duration_by_horizon[horizon] = duration
    if max(
        by_lead[["mae_w", "peak_mae_w", "ongoing_peak_mae_w"]].max().max()
        for by_lead in by_lead_by_horizon.values()
    ) > MAE_PLOT_LIMIT_W:
        raise ValueError("MAE exceeds the fixed comparison plot limit; increase it for both splits")
    if not transport_rows:
        raise ValueError("No complete forecast windows were available for Wasserstein scores")
    transport_windows = pd.DataFrame(transport_rows)
    transport_summary = summarize_transport(transport_windows)
    transport_plot_limit_min = TRANSPORT_PLOT_LIMIT_MIN
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / f"{split}_predictions.csv.gz", index=False, compression="gzip")
    for horizon in SCORE_HORIZONS:
        suffix = "" if horizon == "2h" else f"_{horizon}"
        metrics_by_horizon[horizon].to_csv(
            output / f"{split}_metrics{suffix}.csv", index=False, float_format="%.6f"
        )
        by_lead_by_horizon[horizon].to_csv(
            output / f"{split}_metrics_by_lead{suffix}.csv", index=False,
            float_format="%.6f"
        )
        duration_by_horizon[horizon].to_csv(
            output / f"{split}_peak_duration{suffix}.csv", index=False
        )
        make_plot(
            by_lead_by_horizon[horizon], output, split, horizon, MAE_PLOT_LIMIT_W
        )
        make_transport_plot(
            transport_summary, output, split, horizon, transport_plot_limit_min
        )
    transport_windows.to_csv(
        output / f"{split}_wasserstein_windows.csv.gz", index=False,
        compression="gzip", float_format="%.6f"
    )
    transport_summary.to_csv(
        output / f"{split}_wasserstein_summary.csv", index=False, float_format="%.6f"
    )
    details = {
        "split": split,
        "start_inclusive": start.isoformat(),
        "end_exclusive": end.isoformat(),
        "interval_minutes": 15,
        "horizon_minutes": 720,
        "score_horizons_minutes": {horizon: steps * 15 for horizon, steps in SCORE_HORIZONS.items()},
        "possible_origins": len(origins),
        "eligible_origins": eligible_origins,
        "possible_12h_origins": len(pd.date_range(start, last_long_origin, freq=INTERVAL)),
        "eligible_12h_origins": eligible_12h_origins,
        "complete_transport_windows_by_horizon": complete_window_origins,
        "eligible_origin_rule": "latest 15-minute bin entirely originally observed",
        "truth_rule": "target within split and all 15 source minutes valid, either observed or approved short-gap interpolation",
        "observed_only_training_rule": "all training labels and feature source values originally observed",
        "ongoing_peak_origins": ongoing_peak_origins,
        "duration_incomplete_origins": duration_incomplete_origins,
        "duration_censored_origins": duration_censored_origins,
        "duration_scored_origins": {
            horizon: len(duration_by_horizon[horizon]) // len(METHODS)
            for horizon in SCORE_HORIZONS
        },
        "duration_score_rule": "run of peak target bins capped at each 2h or 12h horizon; missing before observed end excluded",
        "transport_rule": "one-dimensional Wasserstein-1 over 15-minute time locations, using power times 0.25h as Wh weights; each positive-mass profile independently normalized; result is minutes",
        "peak_transport_rule": "weights are positive excess over fixed peak threshold times 0.25h; distance undefined if either profile has zero peak-excess mass",
        "peak_energy_rule": "missing=max(actual excess Wh minus forecast excess Wh,0); extra is reverse; peak-height error is forecast horizon max minus realized horizon max",
        "transport_truth_rule": "all 8 or 48 target bins complete and approved; exclude incomplete windows rather than filling long gaps",
        "transport_plot_limit_min": transport_plot_limit_min,
        "peak_threshold_w": threshold_w,
        "peak_threshold_rule": "95th percentile of fully observed 15-minute bins, August 2-21, 2026",
        "p95_rule": "95th percentile for each time-of-day slot over prior 14 or 20 completed days, linear quantile interpolation",
        "p95_history": "complete approved bins; last hour before daily issue excludes interpolated bins whose gap may close after issue",
        "mean_rule": "arithmetic mean for each time-of-day slot over prior 14 or 20 completed days",
        "mean_history": "complete approved bins; last hour before daily issue excludes interpolated bins whose gap may close after issue",
        "forest": model_details,
        "rf_augmented": augmented_details,
        "arx_1": arx_details,
        "arima_101": arima_details,
        "arima_202": arima_202_details,
        "rf_q75_peak": {
            "formula": "B_k=max(0, RF20_k + max(0, q75_residual_k)); if latest<thresh, forecast_k=B_k; otherwise forecast_k=B_k+max(latest-B_1,0)*S_age(k)",
            "causal_issue_rule": "latest and observed run age use only originally observed bins before issue; unknown onset uses pooled continuation",
            "calibration": q75_details,
            "continuation": continuation_details,
        },
        "source": str(data.resolve()),
        "source_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "source_details": source_details,
        "mae_plot_limit_w": MAE_PLOT_LIMIT_W,
    }
    (output / f"{split}_details.json").write_text(json.dumps(details, indent=2) + "\n")
    for horizon in SCORE_HORIZONS:
        print(f"\n{horizon} pointwise and duration metrics")
        print(metrics_by_horizon[horizon].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\nPeak threshold: {threshold_w:.1f} W")
    print(f"Eligible 2h origins: {eligible_origins}/{len(origins)}; "
          f"eligible 12h origins: {eligible_12h_origins}/{details['possible_12h_origins']}")
    print(f"Complete Wasserstein windows: {complete_window_origins}; "
          f"duration-scored peak origins: {details['duration_scored_origins']}")
    print(f"Saved results to {output}")
    return details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_evaluation(
        data=args.data,
        output=args.output,
        start=pd.Timestamp("2026-08-22"),
        end=pd.Timestamp("2026-09-05"),
        split="validation",
    )


if __name__ == "__main__":
    main()
