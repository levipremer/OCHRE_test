"""Compare causal, EV-only Home Hub forecasts on six baseline simulation weeks.

Input rows are 15-minute *interval starts*. The controlled quantity is the
non-EV Hub mean: the measured Hub meter minus the separately metered EV. With
observed ambient temperature, this module computes fold-specific QCP heating
features from completed source hours; precomputed global QCP columns are
rejected because their fitting cutoff cannot be verified for each fold.

Weeks 1-4 precede validation, week 5 supplies four rolling, purged daily
folds, and week 6 is held out. For each fold and final test fit, a model uses
exactly the preceding 2, 3, or 4 weeks. A training origin is retained only
when its target interval has ended by the fit cutoff. Model selection uses
validation MAE only. Test labels are read only by the scoring stage.

The established 20-day conservative RF + 14/20-day p90 hybrid is evaluated as
a separate reference. Its q75 residual calibration, peak continuation, and
two-hour handoff are imported unchanged from the rest-of-home experiment.
Alternative handoffs and percentiles are diagnostics, never deployed here.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .evaluate_rest_of_home import (
        conservative_peak_forecast,
        features_from_observed,
        fit_forest,
        fit_nonpeak_residual_quantiles,
        fit_peak_continuation,
        observed_peak_run_age,
    )
    from .qcp_heating_feature import (
        features_at_origins,
        hourly_from_baseline,
        try_fit_hour_pooled_qcp,
        walk_forward_training_features,
    )
except ImportError:  # direct script invocation
    from evaluate_rest_of_home import (
        conservative_peak_forecast,
        features_from_observed,
        fit_forest,
        fit_nonpeak_residual_quantiles,
        fit_peak_continuation,
        observed_peak_run_age,
    )
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from examples.single_home_hems.forecasting.qcp_heating_feature import (
        features_at_origins,
        hourly_from_baseline,
        try_fit_hour_pooled_qcp,
        walk_forward_training_features,
    )


INTERVAL = pd.Timedelta(minutes=15)
DAY = pd.Timedelta(days=1)
WEEK = pd.Timedelta(days=7)
HORIZON_STEPS = 32  # EV-only MPC's eight-hour prediction horizon.
TRAIN_WEEKS = (2, 3, 4)
CV_DAY_OFFSETS = (0, 2, 4, 6)
PERCENTILES = (50.0, 75.0, 80.0, 85.0, 90.0, 95.0)
P90_HISTORY_DAYS = (14, 20)
HANDOFF_STEPS = (4, 8, 12, 16)
REFERENCE_HANDOFF_STEPS = 8
SEED = 42
CALENDAR_COLUMNS = (
    "lead_step", "target_clock_sin", "target_clock_cos", "target_week_sin",
    "target_week_cos", "target_is_weekend",
)
QCP_COLUMNS = ("qcp_heating_feature_kw", "qcp_feature_available")


@dataclass(frozen=True)
class StudyRows:
    x: pd.DataFrame
    y: np.ndarray
    origin: pd.DatetimeIndex
    target: pd.DatetimeIndex
    lead: np.ndarray
    feature_sets: dict[str, tuple[str, ...]]

    def mask_fit(self, start: pd.Timestamp, cutoff: pd.Timestamp,
                 origin_stride: int = 4) -> np.ndarray:
        """Keep hourly origins with the full eight-hour label window complete."""
        return np.asarray(
            (self.origin >= start) & (self.origin < cutoff)
            & (self.origin + HORIZON_STEPS * INTERVAL <= cutoff)
            & ((self.origin.hour * 4 + self.origin.minute // 15) % origin_stride == 0)
            & np.isfinite(self.y),
            dtype=bool,
        )

    def mask_evaluation(self, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
        """Score only origins with a complete horizon inside the split."""
        return np.asarray(
            (self.origin >= start) & (self.origin + HORIZON_STEPS * INTERVAL <= end)
            & np.isfinite(self.y),
            dtype=bool,
        )


def read_baseline(path: Path, weather_minute_path: Path | None = None,
                  weather_15min_path: Path | None = None) -> pd.DataFrame:
    """Validate one complete six-week 15-minute baseline and its meter balance."""
    raw = pd.read_csv(path)
    needed = {"timestamp", "home_hub_power_kw", "ev_power_kw", "hub_non_ev_power_kw"}
    missing = needed - set(raw)
    if missing:
        raise ValueError(f"Simulated Hub baseline lacks {sorted(missing)}")
    time = pd.to_datetime(raw["timestamp"], errors="raise")
    if time.dt.tz is not None:
        # OCHRE's local wall clock is the modeling clock; retain its local zone.
        time = time.dt.tz_localize(None)
    index = pd.DatetimeIndex(time, name="timestamp")
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("Baseline interval starts must be unique and increasing")
    if len(index) != 6 * 7 * 96 or index[0] != index[0].normalize():
        raise ValueError("Baseline must contain exactly six complete weeks from local midnight")
    expected = pd.date_range(index[0], periods=len(index), freq=INTERVAL, name="timestamp")
    if not index.equals(expected):
        raise ValueError("Baseline must have every 15-minute interval in the six weeks")
    raw.index = index
    for column in needed - {"timestamp"}:
        raw[column] = pd.to_numeric(raw[column], errors="raise")
    power = raw[["home_hub_power_kw", "ev_power_kw", "hub_non_ev_power_kw"]].to_numpy(float)
    if not np.isfinite(power).all():
        raise ValueError("Baseline power contains missing or nonfinite values")
    if (power[:, 2] < -1e-7).any() or (power[:, 0] < -1e-7).any():
        raise ValueError("Hub and non-EV Hub demand must be nonnegative")
    if not np.allclose(power[:, 0] - power[:, 1], power[:, 2], rtol=0, atol=1e-5):
        raise ValueError("Hub minus EV meter does not equal the non-EV target")
    if "qcp_heating_feature_kw" in raw or "qcp_feature_ready_at" in raw:
        raise ValueError("QCP features must be generated within each chronological fit fold")
    if weather_minute_path is not None and weather_15min_path is not None:
        raise ValueError("Choose only one realized weather input")
    if weather_minute_path is not None or weather_15min_path is not None:
        if "ambient_temperature_c" in raw:
            raise ValueError("Baseline already contains ambient temperature; omit separate weather input")
        weather = pd.read_csv(weather_minute_path or weather_15min_path)
        if not {"timestamp", "ambient_temperature_c"}.issubset(weather):
            raise ValueError("Weather CSV needs timestamp and ambient_temperature_c")
        minute_time = pd.to_datetime(weather["timestamp"], errors="raise")
        if minute_time.dt.tz is not None:
            minute_time = minute_time.dt.tz_localize(None)
        minute_index = pd.DatetimeIndex(minute_time)
        if minute_index.has_duplicates or not minute_index.is_monotonic_increasing:
            raise ValueError("Weather minute timestamps must be unique and increasing")
        temperature = pd.Series(pd.to_numeric(weather["ambient_temperature_c"],
                                              errors="raise").to_numpy(float),
                                index=minute_index)
        if weather_15min_path is not None:
            selected = temperature.reindex(index)
            if len(temperature) != len(index) or not minute_index.equals(index):
                raise ValueError("15-minute weather must match every Hub interval start")
            if not np.isfinite(selected.to_numpy(float)).all():
                raise ValueError("15-minute weather contains missing or nonfinite values")
            raw["ambient_temperature_c"] = selected.to_numpy(float)
        else:
            minutes = temperature.loc[(temperature.index >= index[0])
                                      & (temperature.index < index[-1] + INTERVAL)]
            if len(minutes) != len(index) * 15 or not minutes.index.equals(
                pd.date_range(index[0], periods=len(index) * 15, freq="min")
            ) or not np.isfinite(minutes.to_numpy(float)).all():
                raise ValueError("Weather needs every observed minute of the six-week baseline")
            grouped = minutes.resample("15min", closed="left", label="left")
            if not grouped.count().reindex(index).eq(15).all():
                raise ValueError("Each Hub interval needs 15 weather measurements")
            raw["ambient_temperature_c"] = grouped.mean().reindex(index).to_numpy(float)
    return raw


def make_supervised(data: pd.DataFrame) -> StudyRows:
    """Use only completed aggregate Hub measurements in origin features."""
    values = data["hub_non_ev_power_kw"].to_numpy(float)
    n = len(data)
    index = data.index
    history = pd.DataFrame(index=index)
    series = pd.Series(values, index=index)
    for lag in (1, 2, 4, 8, 96, 672):
        lagged = series.shift(lag)
        history[f"load_lag_{lag}_kw"] = lagged.fillna(0.0)
        history[f"load_lag_{lag}_available"] = lagged.notna().astype(int)
    previous = series.shift(1)
    for steps in (4, 8, 96):
        rolling = previous.rolling(steps, min_periods=1)
        history[f"load_mean_prev_{steps}_kw"] = rolling.mean().fillna(0.0)
        history[f"load_max_prev_{steps}_kw"] = rolling.max().fillna(0.0)
        history[f"load_count_prev_{steps}"] = rolling.count().astype(int)
    prior_hour = series.shift(5)
    slope = previous - prior_hour
    history["load_slope_1h_kw"] = slope.fillna(0.0)
    history["load_slope_1h_available"] = slope.notna().astype(int)
    if "ambient_temperature_c" in data:
        # At issue t, the latest available ambient reading is t-15 minutes.
        ambient = pd.to_numeric(
            data["ambient_temperature_c"], errors="coerce"
        ).shift(1)
        history["ambient_lag_1_c"] = ambient.fillna(0.0)
        history["ambient_lag_1_available"] = ambient.notna().astype(int)
    origins = np.repeat(np.arange(1, n - HORIZON_STEPS + 1), HORIZON_STEPS)
    leads = np.tile(np.arange(1, HORIZON_STEPS + 1), len(origins) // HORIZON_STEPS)
    targets = origins + leads - 1
    origin_times = index.take(origins)
    target_times = index.take(targets)
    x = history.iloc[origins].reset_index(drop=True)
    x["lead_step"] = leads
    clock = 2 * np.pi * (target_times.hour * 60 + target_times.minute) / 1440
    week = 2 * np.pi * target_times.dayofweek / 7
    x["target_clock_sin"] = np.sin(clock)
    x["target_clock_cos"] = np.cos(clock)
    x["target_week_sin"] = np.sin(week)
    x["target_week_cos"] = np.cos(week)
    x["target_is_weekend"] = (target_times.dayofweek >= 5).astype(int)
    history_columns = tuple(history)
    feature_sets = {
        "calendar": CALENDAR_COLUMNS,
        "history": CALENDAR_COLUMNS + history_columns,
    }
    if not np.isfinite(x.loc[:, list(feature_sets["history"])].to_numpy(float)).all():
        raise ValueError("Historical Hub feature matrix contains nonfinite values")
    return StudyRows(
        x=x, y=values[targets], origin=origin_times, target=target_times,
        lead=leads, feature_sets=feature_sets,
    )


def _candidate_specs(has_qcp: bool) -> dict[str, list[dict[str, Any]]]:
    specs: dict[str, list[dict[str, Any]]] = {
        "ridge_calendar": [{"alpha": a} for a in (1.0, 100.0)],
        "ridge_history": [{"alpha": a} for a in (1.0, 100.0)],
        "rf_history": [{"min_samples_leaf": leaf} for leaf in (5, 15)],
        "hgb_history": [{"max_leaf_nodes": leaves} for leaves in (15, 31)],
    }
    if has_qcp:
        specs["rf_history_qcp"] = [{"min_samples_leaf": leaf} for leaf in (5, 15)]
    return specs


def _feature_set(method: str) -> str:
    return "history_qcp" if method.endswith("_qcp") else (
        "calendar" if method.endswith("_calendar") else "history"
    )


def _model(method: str, params: dict[str, Any]):
    if method.startswith("ridge_"):
        return make_pipeline(StandardScaler(), Ridge(**params))
    if method.startswith("rf_"):
        return RandomForestRegressor(
            n_estimators=80, max_features=0.8, max_samples=0.8,
            bootstrap=True, n_jobs=4, random_state=SEED, **params,
        )
    if method.startswith("hgb_"):
        return HistGradientBoostingRegressor(
            max_iter=120, learning_rate=0.06, l2_regularization=1.0,
            early_stopping=False, random_state=SEED, **params,
        )
    raise ValueError(f"Unknown model {method}")


def _predict(model: Any, rows: StudyRows, mask: np.ndarray, columns: tuple[str, ...]) -> np.ndarray:
    result = np.asarray(model.predict(rows.x.loc[mask, list(columns)]), dtype=float)
    if not np.isfinite(result).all():
        raise ValueError("Forecast model returned nonfinite power")
    return np.maximum(result, 0.0)


def _qcp_matrix(rows: StudyRows, mask: np.ndarray,
                issued_features: pd.DataFrame) -> pd.DataFrame:
    """Join a fold-specific QCP issue to origin rows, retaining readiness."""
    if tuple(issued_features.columns) != ("qcp_heating_feature_kw", "qcp_feature_ready_at"):
        raise ValueError("Issued QCP feature has an unexpected schema")
    origins = pd.DatetimeIndex(rows.origin[mask], name="timestamp")
    aligned = issued_features.reindex(origins)
    ready = pd.to_datetime(aligned["qcp_feature_ready_at"])
    available = aligned["qcp_heating_feature_kw"].notna().to_numpy()
    if ready.loc[available].isna().any() or (ready.loc[available].to_numpy() > origins[available].to_numpy()).any():
        raise ValueError("QCP feature was not complete when its forecast was issued")
    x = rows.x.loc[mask, list(rows.feature_sets["history"])].copy()
    x[QCP_COLUMNS[0]] = aligned["qcp_heating_feature_kw"].fillna(0.0).to_numpy(float)
    x[QCP_COLUMNS[1]] = available.astype(int)
    return x


def _fold_qcp_matrices(hourly: pd.DataFrame, rows: StudyRows,
                       fit_mask: np.ndarray, valid_mask: np.ndarray,
                       train_start: pd.Timestamp, cutoff: pd.Timestamp
                       ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Fit QCP inside one chronological fold, with causal training features."""
    training_origins = pd.DatetimeIndex(rows.origin[fit_mask].unique())
    validation_origins = pd.DatetimeIndex(rows.origin[valid_mask].unique())
    training_features, training_info = walk_forward_training_features(
        hourly, training_origins, train_start
    )
    model, coverage = try_fit_hour_pooled_qcp(hourly, train_start, cutoff)
    validation_features = features_at_origins(model, hourly, validation_origins)
    return (
        _qcp_matrix(rows, fit_mask, training_features),
        _qcp_matrix(rows, valid_mask, validation_features),
        {"training_feature_info": training_info, "evaluation_fit": coverage},
    )


def _prediction_rows(rows: StudyRows, mask: np.ndarray, forecast: np.ndarray,
                     method: str, train_weeks: int, split: str) -> pd.DataFrame:
    if len(forecast) != int(mask.sum()):
        raise ValueError("Prediction count differs from selected origin-target pairs")
    return pd.DataFrame({
        "method": method, "train_weeks": train_weeks, "split": split,
        "origin": rows.origin[mask], "target": rows.target[mask],
        "lead": rows.lead[mask], "forecast_kw": forecast,
        "actual_kw": rows.y[mask],
    })


def _score(frame: pd.DataFrame, threshold_kw: float) -> dict[str, float | int]:
    error = frame["forecast_kw"].to_numpy(float) - frame["actual_kw"].to_numpy(float)
    forecast_peak = frame["forecast_kw"].to_numpy(float) >= threshold_kw
    actual_peak = frame["actual_kw"].to_numpy(float) >= threshold_kw
    nonpeak = ~actual_peak
    return {
        "pairs": len(frame), "origins": frame["origin"].nunique(),
        "mae_kw": float(np.abs(error).mean()),
        "rmse_kw": float(np.sqrt(np.square(error).mean())),
        "bias_kw": float(error.mean()),
        "p95_absolute_error_kw": float(np.quantile(np.abs(error), 0.95)),
        "underforecast_rate": float(np.mean(error < 0)),
        "peak_recall": float(forecast_peak[actual_peak].mean()) if actual_peak.any() else np.nan,
        "peak_precision": float(actual_peak[forecast_peak].mean()) if forecast_peak.any() else np.nan,
        "actual_peak_mean_shortfall_kw": (
            float(np.maximum(-error[actual_peak], 0).mean()) if actual_peak.any() else np.nan
        ),
        "nonpeak_bias_kw": float(error[nonpeak].mean()) if nonpeak.any() else np.nan,
    }


def score_predictions(predictions: pd.DataFrame, threshold_kw: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = []
    leads = []
    for (method, train_weeks, split), group in predictions.groupby(
        ["method", "train_weeks", "split"], sort=False, dropna=False
    ):
        for hours, steps in ((2, 8), (4, 16), (8, 32)):
            selected = group.loc[group["lead"] <= steps]
            if not selected.empty:
                summary.append({
                    "method": method, "train_weeks": train_weeks, "split": split,
                    "horizon_hours": hours, "scope": "through_horizon",
                    **_score(selected, threshold_kw),
                })
            at = group.loc[group["lead"] == steps]
            if not at.empty:
                summary.append({
                    "method": method, "train_weeks": train_weeks, "split": split,
                    "horizon_hours": hours, "scope": "at_horizon",
                    **_score(at, threshold_kw),
                })
        for lead, selected in group.groupby("lead", sort=True):
            leads.append({
                "method": method, "train_weeks": train_weeks, "split": split,
                "lead": lead, "lead_minutes": 15 * lead,
                **_score(selected, threshold_kw),
            })
    return pd.DataFrame(summary), pd.DataFrame(leads)


def score_by_issue_day(predictions: pd.DataFrame, threshold_kw: float) -> pd.DataFrame:
    """Preserve issue-day units for paired uncertainty analysis of overlapping horizons."""
    scored = predictions.copy()
    scored["issue_day"] = pd.to_datetime(scored["origin"]).dt.normalize()
    records = []
    for (method, train_weeks, split, day), group in scored.groupby(
        ["method", "train_weeks", "split", "issue_day"], sort=False, dropna=False
    ):
        records.append({
            "method": method, "train_weeks": train_weeks, "split": split,
            "issue_day": day, **_score(group, threshold_kw),
        })
    return pd.DataFrame(records)


def plot_comparison_diagnostics(lead: pd.DataFrame, sensitivity: pd.DataFrame,
                                selected: dict[str, dict[str, Any]],
                                output_dir: Path) -> None:
    """Plot validation diagnostics and locked test choices on consistent axes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    validation_names = (
        "conservative_rf_all_leads", "clock_p90_14d", "clock_p90_20d",
        "hybrid_rf_p90_14d_120min", "hybrid_rf_p90_20d_120min",
    )
    validation = lead.loc[
        lead["split"].eq("validation") & lead["method"].isin(validation_names)
    ]
    selected_names = []
    for weeks in TRAIN_WEEKS:
        choices = [(key.split(":", 1)[1], item["cv_mae_kw"])
                   for key, item in selected.items() if key.startswith(f"{weeks}w:")]
        best = min(choices, key=lambda item: (item[1], item[0]))[0]
        selected_names.append((weeks, best))
    test_parts = []
    for weeks, method in selected_names:
        part = lead.loc[
            lead["split"].eq("test") & lead["train_weeks"].eq(weeks)
            & lead["method"].eq(method)
        ].copy()
        part["label"] = f"{weeks} weeks · {method}"
        test_parts.append(part)
    for method in ("hybrid_rf_p90_14d_120min", "hybrid_rf_p90_20d_120min"):
        part = lead.loc[lead["split"].eq("test") & lead["method"].eq(method)].copy()
        part["label"] = method
        if not part.empty:
            test_parts.append(part)
    test_curves = pd.concat(test_parts, ignore_index=True)

    plotted_mae = np.concatenate((
        validation["mae_kw"].to_numpy(float),
        test_curves["mae_kw"].to_numpy(float),
        sensitivity["mae_kw"].to_numpy(float),
    ))
    mae_ymax = max(0.1, float(np.nanmax(plotted_mae)) * 1.08)
    plotted_bias = np.concatenate((
        validation["bias_kw"].to_numpy(float), test_curves["bias_kw"].to_numpy(float)
    ))
    bias_limit = max(0.1, float(np.nanmax(np.abs(plotted_bias))) * 1.08)
    for metric, ylabel, limits, filename in (
        ("mae_kw", "MAE (kW)", (0, mae_ymax), "lead_mae_validation_test.png"),
        ("bias_kw", "Mean forecast minus actual (kW)",
         (-bias_limit, bias_limit), "lead_bias_validation_test.png"),
    ):
        figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True,
                                    layout="constrained")
        for method, part in validation.groupby("method", sort=False):
            axes[0].plot(part["lead_minutes"], part[metric], label=method)
        for label, part in test_curves.groupby("label", sort=False):
            axes[1].plot(part["lead_minutes"], part[metric], label=label)
        for ax, title in zip(axes, ("Week 5 validation · reference diagnostics",
                                    "Week 6 held-out · validation-selected models")):
            ax.set(title=title, xlabel="Forecast lead (minutes)", ylabel=ylabel,
                   xlim=(15, HORIZON_STEPS * 15), ylim=limits)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
        figure.savefig(output_dir / filename, dpi=160)
        plt.close(figure)

    direct = lead.loc[lead["split"].eq("test") & lead["train_weeks"].notna()]
    if not direct.empty:
        means = direct.groupby(["method", "train_weeks"], sort=True)["mae_kw"].mean().unstack()
        figure, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
        positions = np.arange(len(means))
        width = 0.25
        for offset, weeks in zip((-1, 0, 1), TRAIN_WEEKS):
            ax.bar(positions + offset * width, means[weeks], width=width,
                   label=f"{weeks} weeks")
        ax.set(ylabel="Eight-hour MAE (kW)", ylim=(0, mae_ymax),
               xticks=positions, xticklabels=means.index,
               title="Week 6 held-out · models trained on the last 2, 3, or 4 weeks")
        ax.tick_params(axis="x", labelrotation=25)
        ax.grid(axis="y", alpha=0.2)
        ax.legend()
        figure.savefig(output_dir / "test_model_training_window_mae.png", dpi=160)
        plt.close(figure)

    if sensitivity.empty:
        return
    sensitivity = sensitivity.loc[sensitivity["split"].eq("validation")]
    for kind, fixed, x, filename in (
        ("percentile", sensitivity.loc[sensitivity["handoff_minutes"].eq(120)],
         "percentile", "validation_percentile_sensitivity.png"),
        ("handoff", sensitivity.loc[sensitivity["percentile"].eq(90)],
         "handoff_minutes", "validation_handoff_sensitivity.png"),
    ):
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
        for days, part in fixed.groupby("history_days", sort=True):
            part = part.sort_values(x)
            axes[0].plot(part[x], part["mae_kw"], marker="o", label=f"{days} days")
            axes[1].plot(part[x], part["peak_recall"], marker="o", label=f"{days} days")
        if kind == "percentile":
            for ax in axes:
                ax.axvline(90, color="0.5", linestyle=":", linewidth=1)
        else:
            for ax in axes:
                ax.axvline(120, color="0.5", linestyle=":", linewidth=1)
        axes[0].set(ylabel="Eight-hour MAE (kW)", ylim=(0, mae_ymax))
        axes[1].set(ylabel="Peak recall", ylim=(0, 1))
        for ax in axes:
            ax.set(xlabel="Clock percentile" if kind == "percentile" else
                   "Conservative RF handoff (minutes)")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
        figure.suptitle("Week 5 validation · analysis only")
        figure.savefig(output_dir / filename, dpi=160)
        plt.close(figure)


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if np.isfinite(value) else "n/a"


def write_study_report(output_dir: Path, summary: pd.DataFrame, lead: pd.DataFrame,
                       sensitivity: pd.DataFrame, qcp_coverage: pd.DataFrame,
                       selected: dict[str, dict[str, Any]]) -> None:
    """Summarize validation-selected methods and untouched held-out scores."""
    lines = [
        "# Six-week simulated Home Hub forecast results", "",
        "The target is 15-minute mean non-EV Hub power. Week 5 selects model settings; "
        "week 6 is held out for scoring. Error units are kW. The EV controller's "
        "three eligible-load forecast channels are not produced by this study.", "",
        "## Held-out test: validation-selected model for each training window", "",
        "| Training window | Model chosen on week 5 | 2h MAE | 2h bias | 8h MAE | 8h bias |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for weeks in TRAIN_WEEKS:
        choices = [(key.split(":", 1)[1], item["cv_mae_kw"])
                   for key, item in selected.items() if key.startswith(f"{weeks}w:")]
        method = min(choices, key=lambda item: (item[1], item[0]))[0]
        scores = summary.loc[
            summary["split"].eq("test") & summary["train_weeks"].eq(weeks)
            & summary["method"].eq(method) & summary["scope"].eq("through_horizon")
        ].set_index("horizon_hours")
        lines.append(
            f"| {weeks} weeks | {method} | {_fmt(scores.loc[2, 'mae_kw'])} | "
            f"{_fmt(scores.loc[2, 'bias_kw'])} | {_fmt(scores.loc[8, 'mae_kw'])} | "
            f"{_fmt(scores.loc[8, 'bias_kw'])} |"
        )
    lines += [
        "", "Every candidate and its 2/3/4-week test result appears in "
        "[horizon_scores.csv](horizon_scores.csv); the table above chooses models "
        "using validation only.", "",
        "## Conservative RF and clock p90", "",
        "The original RF/q75/peak-continuation calibration uses 20 days; it is a "
        "separate reference rather than a 2/3/4-week fit. Its two-hour handoff "
        "and p90 are unchanged. The reference can update its clock profile from "
        "completed earlier test days.", "",
    ]
    for days in P90_HISTORY_DAYS:
        method = f"hybrid_rf_p90_{days}d_120min"
        score = summary.loc[
            summary["split"].eq("test") & summary["method"].eq(method)
            & summary["scope"].eq("through_horizon")
            & summary["horizon_hours"].eq(8)
        ]
        if not score.empty:
            row = score.iloc[0]
            lines.append(
                f"- {days}-day clock history: test 8h MAE {_fmt(row['mae_kw'])} kW, "
                f"bias {_fmt(row['bias_kw'])} kW, peak recall {_fmt(row['peak_recall'])}."
            )
    lines += ["", "Validation RF versus p90 by lead:", ""]
    reference = lead.loc[
        lead["split"].eq("validation") & lead["method"].isin((
            "conservative_rf_all_leads", "clock_p90_14d", "clock_p90_20d"
        ))
    ].pivot(index="lead", columns="method", values="mae_kw")
    if not reference.empty:
        rf = reference["conservative_rf_all_leads"]
        for days in P90_HISTORY_DAYS:
            better = reference[f"clock_p90_{days}d"] < rf
            first = int(better[better].index[0]) if better.any() else None
            lines.append(
                f"- {days}-day p90 beats RF on {int(better.sum())}/{len(better)} leads; "
                f"first such lead: {first * 15 if first is not None else 'none'} minutes."
            )
    lines += [
        "", "An isolated better lead would not establish a stable crossover. See "
        "[the lead curve](lead_mae_validation_test.png) and "
        "[per-lead values](rf_vs_p90_by_lead_validation.csv).", "",
        "## Validation tradeoff for the established p90 reference", "",
        "The nearby clock percentiles and handoffs are evaluated on week 5 only. "
        "The p90 setting remains the established reference. Validation does "
        "not favor it by MAE; it captures more peaks at the cost of positive "
        "bias. Changing that forecast formulation requires user confirmation, "
        "and week 6 is used only for scoring.", "",
        "| Clock days | Percentile | 8h MAE | Peak recall | Nonpeak bias |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    fixed = sensitivity.loc[sensitivity["handoff_minutes"].eq(120)]
    for row in fixed.sort_values(["history_days", "percentile"]).itertuples():
        lines.append(
            f"| {row.history_days} | {row.percentile:g} | {_fmt(row.mae_kw)} | "
            f"{_fmt(row.peak_recall)} | {_fmt(row.nonpeak_bias_kw)} |"
        )
    lines.append("")
    for days in P90_HISTORY_DAYS:
        group = fixed.loc[fixed["history_days"].eq(days)]
        if group.empty:
            continue
        best = group.loc[group["mae_kw"].idxmin()]
        p90 = group.loc[group["percentile"].eq(90)].iloc[0]
        lines.append(
            f"For {days} days, p{best['percentile']:g} has the lowest validation "
            f"MAE in this grid; p90 is {_fmt(p90['mae_kw'] - best['mae_kw'])} kW "
            f"higher, with peak recall {_fmt(p90['peak_recall'])} versus "
            f"{_fmt(best['peak_recall'])}."
        )
    lines += [
        "", "See [the handoff sweep](validation_handoff_sensitivity.png) and "
        "[all validation diagnostics](percentile_handoff_sensitivity.csv).", "",
        "## QCP feature availability and effect", "",
    ]
    if qcp_coverage.empty:
        lines.append("No observed ambient temperature was supplied, so no QCP variant ran.")
    else:
        final = qcp_coverage.loc[qcp_coverage["phase"].eq("test_fit")]
        for row in final.sort_values("train_weeks").itertuples():
            weeks = int(row.train_weeks)
            available = bool(row.fit_available)
            base = summary.loc[
                summary["split"].eq("test") & summary["train_weeks"].eq(weeks)
                & summary["method"].eq("rf_history")
                & summary["scope"].eq("through_horizon")
                & summary["horizon_hours"].eq(8), "mae_kw"
            ]
            qcp = summary.loc[
                summary["split"].eq("test") & summary["train_weeks"].eq(weeks)
                & summary["method"].eq("rf_history_qcp")
                & summary["scope"].eq("through_horizon")
                & summary["horizon_hours"].eq(8), "mae_kw"
            ]
            effect = (float(qcp.iloc[0] - base.iloc[0]) if not base.empty and not qcp.empty
                      else np.nan)
            lines.append(
                f"- {weeks} weeks: QCP fit {'available' if available else 'unavailable'}; "
                f"walk-forward feature available at {row.walk_forward_available_origins}/"
                f"{row.walk_forward_total_origins} training origins; QCP RF minus "
                f"plain RF test 8h MAE {_fmt(effect)} kW."
            )
    lines += [
        "", "[QCP coverage by fold](qcp_fold_coverage.csv) records sparse "
        "temperature-bin failures. OCHRE thermal submeters are hidden from "
        "every forecast feature; a separate QCP fidelity check may use them "
        "only as evaluation labels.", "",
        "## Files", "",
        "[Held-out model comparison](test_model_training_window_mae.png), "
        "[lead MAE](lead_mae_validation_test.png), "
        "[lead bias](lead_bias_validation_test.png), "
        "[origin-level forecasts](predictions.csv.gz), and "
        "[issue-day scores](issue_day_scores.csv). Overlapping origin-target "
        "pairs are correlated; the day scores preserve an appropriate unit "
        "for paired uncertainty analysis.", "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def train_and_evaluate(data: pd.DataFrame, rows: StudyRows,
                       *, cv_day_offsets: tuple[int, ...] = CV_DAY_OFFSETS,
                       fast: bool = False,
                       only_methods: tuple[str, ...] | None = None,
                       ) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    """Tune on rolling days of week 5; refit on the last N weeks before test."""
    start = data.index[0]
    validation_start = start + 4 * WEEK
    test_start = start + 5 * WEEK
    end = start + 6 * WEEK
    has_qcp = "ambient_temperature_c" in data
    specs = _candidate_specs(has_qcp)
    if only_methods is not None:
        unknown = set(only_methods) - set(specs)
        if unknown or not only_methods or len(set(only_methods)) != len(only_methods):
            raise ValueError(
                f"Model selection must be nonempty, unique, and known: {only_methods}; "
                f"unknown={sorted(unknown)}")
        specs = {method: specs[method] for method in only_methods}
    hourly = hourly_from_baseline(data) if has_qcp else None
    qcp_fold_cache: dict[tuple[int, pd.Timestamp], tuple[pd.DataFrame, pd.DataFrame, dict]] = {}
    if fast:
        specs = {key: values[:1] for key, values in specs.items()}
        cv_day_offsets = cv_day_offsets[:1]
    cv_records = []
    qcp_records = []
    chosen: dict[str, dict[str, Any]] = {}
    test_predictions = []
    test_mask = rows.mask_evaluation(test_start, end)
    for weeks in TRAIN_WEEKS:
        for method, candidates in specs.items():
            is_qcp = method.endswith("_qcp")
            columns = (rows.feature_sets["history"] + QCP_COLUMNS if is_qcp
                       else rows.feature_sets[_feature_set(method)])
            candidate_scores = []
            for candidate_number, params in enumerate(candidates, 1):
                daily_mae = []
                for day_offset in cv_day_offsets:
                    fold_start = validation_start + day_offset * DAY
                    fold_end = fold_start + DAY
                    train_start = fold_start - weeks * WEEK
                    fit = rows.mask_fit(train_start, fold_start)
                    valid = rows.mask_evaluation(fold_start, fold_end)
                    if not fit.any() or not valid.any():
                        raise ValueError(f"Empty {weeks}-week fold at {fold_start}")
                    if is_qcp:
                        key = (weeks, fold_start)
                        if key not in qcp_fold_cache:
                            assert hourly is not None
                            qcp_fold_cache[key] = _fold_qcp_matrices(
                                hourly, rows, fit, valid, train_start, fold_start
                            )
                            qcp_records.append({
                                "phase": "validation", "train_weeks": weeks,
                                "fit_start": train_start, "fit_cutoff": fold_start,
                                **qcp_fold_cache[key][2]["evaluation_fit"],
                                "walk_forward_available_origins": qcp_fold_cache[key][2][
                                    "training_feature_info"
                                ]["available_origins"],
                                "walk_forward_total_origins": qcp_fold_cache[key][2][
                                    "training_feature_info"
                                ]["total_origins"],
                            })
                        x_fit, x_valid, _ = qcp_fold_cache[key]
                    else:
                        x_fit = rows.x.loc[fit, list(columns)]
                        x_valid = rows.x.loc[valid, list(columns)]
                    model = _model(method, params)
                    model.fit(x_fit, rows.y[fit])
                    forecast = np.maximum(np.asarray(model.predict(x_valid), dtype=float), 0.0)
                    mae = float(np.mean(np.abs(forecast - rows.y[valid])))
                    daily_mae.append(mae)
                    cv_records.append({
                        "train_weeks": weeks, "method": method,
                        "candidate": candidate_number, "fold_start": fold_start,
                        "fold_end": fold_end, "fit_start": train_start,
                        "fit_end_exclusive": fold_start, "fit_pairs": int(fit.sum()),
                        "validation_pairs": int(valid.sum()), "mae_kw": mae,
                        "params": json.dumps(params, sort_keys=True),
                    })
                candidate_scores.append(float(np.mean(daily_mae)))
            chosen_params = candidates[int(np.argmin(candidate_scores))]
            chosen[f"{weeks}w:{method}"] = {
                "params": chosen_params, "cv_mae_kw": min(candidate_scores),
            }
            fit = rows.mask_fit(test_start - weeks * WEEK, test_start)
            if is_qcp:
                assert hourly is not None
                x_fit, x_test, qcp_info = _fold_qcp_matrices(
                    hourly, rows, fit, test_mask,
                    test_start - weeks * WEEK, test_start,
                )
                qcp_records.append({
                    "phase": "test_fit", "train_weeks": weeks,
                    "fit_start": test_start - weeks * WEEK, "fit_cutoff": test_start,
                    **qcp_info["evaluation_fit"],
                    "walk_forward_available_origins": qcp_info[
                        "training_feature_info"
                    ]["available_origins"],
                    "walk_forward_total_origins": qcp_info[
                        "training_feature_info"
                    ]["total_origins"],
                })
            else:
                x_fit = rows.x.loc[fit, list(columns)]
                x_test = rows.x.loc[test_mask, list(columns)]
            model = _model(method, chosen_params)
            model.fit(x_fit, rows.y[fit])
            forecast = np.maximum(np.asarray(model.predict(x_test), dtype=float), 0.0)
            test_predictions.append(_prediction_rows(
                rows, test_mask, forecast, method, weeks, "test"
            ))
            print(f"{weeks}w {method}: CV {min(candidate_scores):.3f} kW; test issued", flush=True)
    return (pd.DataFrame(cv_records), pd.concat(test_predictions, ignore_index=True),
            chosen, pd.DataFrame(qcp_records))


def _p_clock_profile(load: pd.Series, origin: pd.Timestamp, days: int,
                     percentile: float) -> np.ndarray:
    cutoff = origin.normalize()
    window = load.loc[(load.index >= cutoff - days * DAY) & (load.index < cutoff)]
    if len(window) != days * 96 or window.isna().any():
        raise ValueError("Clock percentile needs complete, previously measured days")
    slots = window.index.hour * 4 + window.index.minute // 15
    return window.groupby(slots).quantile(percentile / 100.0,
                                         interpolation="linear").reindex(range(96)).to_numpy(float)


def conservative_reference(data: pd.DataFrame, rows: StudyRows,
                           threshold_kw: float, *, split: str,
                           days: tuple[int, ...] = P90_HISTORY_DAYS,
                           percentiles: tuple[float, ...] = PERCENTILES,
                           handoffs: tuple[int, ...] = HANDOFF_STEPS,
                           fast: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score the unchanged 20-day conservative RF and diagnostic p90 grid."""
    start = data.index[0]
    fit_end = start + (4 if split == "validation" else 5) * WEEK
    eval_end = fit_end + WEEK
    observed = data["hub_non_ev_power_kw"] * 1000.0
    threshold_w = threshold_kw * 1000.0
    forest, _ = fit_forest(observed, fit_end, train_days=20)
    q75_offset, _ = fit_nonpeak_residual_quantiles(observed, fit_end, threshold_w)
    continuation, _ = fit_peak_continuation(observed, fit_end, threshold_w)
    mask = rows.mask_evaluation(fit_end, eval_end)
    if fast:
        mask &= rows.origin.minute == 0
    selected_origin = rows.origin[mask]
    unique_origins = pd.DatetimeIndex(selected_origin.unique())
    features = features_from_observed(observed).reindex(unique_origins)
    if not np.isfinite(features.to_numpy(float)).all():
        raise ValueError("Conservative RF lacks a complete previous Hub block")
    x = features.loc[features.index.repeat(HORIZON_STEPS)].reset_index(drop=True)
    x["lead_step"] = np.tile(np.arange(1, HORIZON_STEPS + 1), len(unique_origins))
    base = np.maximum(forest.predict(x).reshape(-1, HORIZON_STEPS), 0.0)
    near = np.empty_like(base)
    for i, origin in enumerate(unique_origins):
        latest = float(features.loc[origin, "latest_w"])
        if latest >= threshold_w:
            age, known = observed_peak_run_age(observed, origin, threshold_w)
            curve = continuation[("age_1" if age == 1 else "age_2plus") if known else "pooled"]
        else:
            curve = continuation["pooled"]
        near[i] = conservative_peak_forecast(
            base[i], q75_offset[:HORIZON_STEPS], latest, threshold_w,
            curve[:HORIZON_STEPS],
        )
    if len(selected_origin) != near.size:
        raise ValueError("Reference origins lack matched complete horizons")
    # Build each profile once per issue day. The current day's readings never
    # enter its clock percentile, even when a forecast is issued late that day.
    p_values: dict[tuple[int, float], np.ndarray] = {}
    for history_days in days:
        for percentile in percentiles:
            cache: dict[pd.Timestamp, np.ndarray] = {}
            matrix = np.empty_like(near)
            for i, origin in enumerate(unique_origins):
                date = origin.normalize()
                if date not in cache:
                    cache[date] = _p_clock_profile(observed, origin, history_days, percentile)
                targets = pd.date_range(origin, periods=HORIZON_STEPS, freq=INTERVAL)
                slots = targets.hour * 4 + targets.minute // 15
                matrix[i] = cache[date][slots]
            p_values[(history_days, percentile)] = matrix

    common = {
        "origin": selected_origin,
        "target": rows.target[mask],
        "lead": rows.lead[mask],
        "actual_kw": rows.y[mask],
    }
    diagnostics = []
    predictions = []
    rf_frame = pd.DataFrame({**common, "forecast_kw": near.ravel() / 1000.0,
                             "method": "conservative_rf_all_leads", "train_weeks": np.nan,
                             "split": split})
    predictions.append(rf_frame)
    for (history_days, percentile), far in p_values.items():
        if percentile == 90.0:
            predictions.append(pd.DataFrame({
                **common, "forecast_kw": far.ravel() / 1000.0,
                "method": f"clock_p90_{history_days}d", "train_weeks": np.nan,
                "split": split,
            }))
        for handoff in handoffs:
            combined = np.where(np.arange(HORIZON_STEPS)[None, :] < handoff, near, far)
            method = f"hybrid_rf_p{percentile:g}_{history_days}d_{handoff * 15}min"
            frame = pd.DataFrame({
                **common, "forecast_kw": combined.ravel() / 1000.0,
                "method": method, "train_weeks": np.nan, "split": split,
            })
            score = _score(frame, threshold_kw)
            diagnostics.append({
                "split": split, "history_days": history_days,
                "percentile": percentile, "handoff_minutes": handoff * 15,
                **score,
            })
            if (percentile == 90.0 and handoff == REFERENCE_HANDOFF_STEPS):
                predictions.append(frame)
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(diagnostics)


def run(input_path: Path, output_dir: Path, *, weather_minute_path: Path | None = None,
        weather_15min_path: Path | None = None,
        fast: bool = False, include_conservative: bool = True,
        only_methods: tuple[str, ...] | None = None) -> None:
    data = read_baseline(input_path, weather_minute_path, weather_15min_path)
    rows = make_supervised(data)
    start = data.index[0]
    # Fixed before week 5; common reference across every method and split.
    threshold_kw = float(data.loc[data.index < start + 20 * DAY,
                                  "hub_non_ev_power_kw"].quantile(0.95))
    cv, predictions, selected, qcp_coverage = train_and_evaluate(
        data, rows, fast=fast, only_methods=only_methods)
    diagnostics = []
    if include_conservative:
        for split in ("validation", "test"):
            reference, grid = conservative_reference(
                data, rows, threshold_kw, split=split,
                # All sensitivity choices are inspected on week 5 only.
                percentiles=PERCENTILES if split == "validation" else (90.0,),
                handoffs=HANDOFF_STEPS if split == "validation" else (REFERENCE_HANDOFF_STEPS,),
                fast=fast,
            )
            predictions = pd.concat([predictions, reference], ignore_index=True)
            if split == "validation":
                diagnostics.append(grid)
    summary, lead = score_predictions(predictions, threshold_kw)
    output_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_dir / "predictions.csv.gz", "wt", encoding="utf-8") as stream:
        predictions.to_csv(stream, index=False)
    cv.to_csv(output_dir / "rolling_cv.csv", index=False)
    if not qcp_coverage.empty:
        qcp_coverage.to_csv(output_dir / "qcp_fold_coverage.csv", index=False)
    summary.to_csv(output_dir / "horizon_scores.csv", index=False)
    lead.to_csv(output_dir / "lead_scores.csv", index=False)
    score_by_issue_day(predictions, threshold_kw).to_csv(
        output_dir / "issue_day_scores.csv", index=False
    )
    if include_conservative:
        reference_leads = lead.loc[
            lead["split"].eq("validation")
            & lead["method"].isin((
                "conservative_rf_all_leads", "clock_p90_14d", "clock_p90_20d"
            )),
            ["method", "lead", "lead_minutes", "mae_kw", "bias_kw",
             "p95_absolute_error_kw", "actual_peak_mean_shortfall_kw"],
        ]
        reference_leads.to_csv(output_dir / "rf_vs_p90_by_lead_validation.csv", index=False)
    sensitivity = pd.concat(diagnostics, ignore_index=True) if diagnostics else pd.DataFrame()
    if not sensitivity.empty:
        sensitivity.to_csv(
            output_dir / "percentile_handoff_sensitivity.csv", index=False
        )
    if include_conservative:
        plot_comparison_diagnostics(lead, sensitivity, selected, output_dir)
        write_study_report(output_dir, summary, lead, sensitivity,
                           qcp_coverage, selected)
    metadata = {
        "input": str(input_path.resolve()),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "weather_minute_input": (str(weather_minute_path.resolve())
                                 if weather_minute_path is not None else None),
        "weather_15min_input": (str(weather_15min_path.resolve())
                                 if weather_15min_path is not None else None),
        "weather_input_sha256": (
            hashlib.sha256((weather_minute_path or weather_15min_path).read_bytes()).hexdigest()
            if weather_minute_path is not None or weather_15min_path is not None else None
        ),
        "interval_minutes": 15,
        "horizon_steps": HORIZON_STEPS,
        "horizon_hours": 8,
        "baseline_start": start.isoformat(),
        "validation_start": (start + 4 * WEEK).isoformat(),
        "test_start": (start + 5 * WEEK).isoformat(),
        "test_end": (start + 6 * WEEK).isoformat(),
        "training_weeks": TRAIN_WEEKS,
        "evaluated_methods": (
            list(only_methods) if only_methods is not None
            else list(_candidate_specs("ambient_temperature_c" in data))
        ),
        "cv_fold_day_offsets_in_week5": CV_DAY_OFFSETS[:1] if fast else CV_DAY_OFFSETS,
        "cv_rule": "rolling preceding N weeks, purged labels, full-horizon daily validation",
        "fit_origin_stride_minutes": 60,
        "test_origin_stride_minutes": 15,
        "target": "15-minute mean non-EV Hub power = total Hub meter - EV meter",
        "controller_ready": False,
        "controller_readiness_limit": (
            "This study predicts only mean non-EV Hub load. The EV controller "
            "also requires a minimum eligible-load forecast and consistency "
            "checks. Depending on Hub wiring, eligible mean may equal Hub mean "
            "or require a separate forecast."
        ),
        "input_visibility": "completed aggregate Hub and EV meter; optional prior ambient and issued causal QCP feature",
        "private_hvac_hpwh_labels_used_for_training": False,
        "private_eligible_load_labels_used_for_training": False,
        "qcp_feature_included": "ambient_temperature_c" in data,
        "qcp_feature_rule": (
            "QCP refitted inside each 2/3/4-week fold; train-origin features use "
            "walk-forward fits ending at prior midnights, and validation/test "
            "features use a fit frozen at the split boundary"
            if "ambient_temperature_c" in data else None
        ),
        "peak_threshold_kw": threshold_kw,
        "peak_threshold_rule": "95th percentile of the first 20 days, fixed before validation",
        "conservative_reference": (
            "unchanged 20-day RF/q75/continuation with 2h switch and 14/20-day p90"
            if include_conservative else None
        ),
        "reference_online_history_rule": (
            "RF and calibration frozen at each split start; clock-slot p90 refreshes "
            "from completed calendar days before each issue day, including already "
            "observed test days"
            if include_conservative else None
        ),
        "percentile_and_handoff_grid": "analysis only; does not change the reference formulation",
        "selected_validation_params": selected,
        "package_versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__, "pandas": pd.__version__,
            "scikit_learn": __import__("sklearn").__version__,
        },
        "fast": fast,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved benchmark to {output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Six-week, 15-minute baseline meter CSV")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weather-minute", type=Path,
                        help="Optional complete realized weather minute CSV; joined as past observations")
    parser.add_argument("--weather-15min", type=Path,
                        help="Optional aligned 15-minute realized ambient temperature CSV")
    parser.add_argument("--fast", action="store_true",
                        help="One validation fold/candidate and hourly reference origins for a smoke run")
    parser.add_argument("--skip-conservative", action="store_true",
                        help="Run only new 2/3/4-week models")
    parser.add_argument("--only-method", action="append",
                        help="Evaluate only this established direct method; repeat to select more")
    args = parser.parse_args()
    run(args.input, args.output_dir, weather_minute_path=args.weather_minute,
        weather_15min_path=args.weather_15min,
        fast=args.fast,
        include_conservative=not args.skip_conservative,
        only_methods=tuple(args.only_method) if args.only_method else None)


if __name__ == "__main__":
    main()
