"""Feature-augmented, causal 12-hour rest-of-home random-forest forecast.

All power features use originally observed 15-minute bins ending before the
forecast origin.  Calendar features for the target are known at issue time.
This is an additional candidate; the original forest remains unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


INTERVAL = pd.Timedelta(minutes=15)
TRAIN_DAYS = 20
HORIZON_STEPS = 48
FOREST_TREES = 100
FOREST_MIN_SAMPLES_LEAF = 3
FOREST_SEED = 42


def origin_features(observed: pd.Series, threshold_w: float) -> pd.DataFrame:
    """Build features at t from originally observed bins strictly before t.

    The latest observed bin starts at t-15 minutes. Missing earlier bins do
    not remove an otherwise eligible origin: slopes are zero when unavailable,
    with a separate validity flag for each slope. Rolling summaries ignore
    missing bins and retain their observed-count features.
    """
    previous = observed.shift(1)
    recent_30m = previous.rolling(2, min_periods=1)
    recent_1h = previous.rolling(4, min_periods=1)
    recent_2h = previous.rolling(8, min_periods=1)
    clock_angle = 2 * np.pi * (
        observed.index.hour * 60 + observed.index.minute
    ) / 1440
    week_angle = 2 * np.pi * observed.index.dayofweek / 7

    # The one-hour slope compares the latest completed bin with the bin one
    # hour earlier, at t-75 minutes. A missing comparison is distinct from a
    # measured zero slope.
    prior_15m = observed.shift(2)
    prior_1h = observed.shift(5)
    slope_15m_valid = previous.notna() & prior_15m.notna()
    slope_1h_valid = previous.notna() & prior_1h.notna()
    observed_peak = previous.ge(threshold_w).astype(float).where(previous.notna())

    return pd.DataFrame(
        {
            # The existing forest's origin features are preserved verbatim.
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
            # Approved additions.
            "slope_15m_w": (previous - prior_15m).where(slope_15m_valid, 0.0),
            "slope_15m_valid": slope_15m_valid.astype(int),
            "slope_1h_w": (previous - prior_1h).where(slope_1h_valid, 0.0),
            "slope_1h_valid": slope_1h_valid.astype(int),
            "observed_peak_count_2h": observed_peak.rolling(8, min_periods=1)
            .sum().fillna(0.0),
        },
        index=observed.index,
    )


def target_features(origins: pd.DatetimeIndex, lead: int) -> pd.DataFrame:
    """Known target clock/weekday features for one future lead."""
    targets = origins + (lead - 1) * INTERVAL
    clock_angle = 2 * np.pi * (targets.hour * 60 + targets.minute) / 1440
    week_angle = 2 * np.pi * targets.dayofweek / 7
    return pd.DataFrame(
        {
            "target_clock_sin": np.sin(clock_angle),
            "target_clock_cos": np.cos(clock_angle),
            "target_week_sin": np.sin(week_angle),
            "target_week_cos": np.cos(week_angle),
            "lead_step": lead,
        },
        index=origins,
    )


def fit_augmented_forest(
    observed: pd.Series,
    train_end: pd.Timestamp,
    threshold_w: float,
    *,
    train_days: int = TRAIN_DAYS,
    steps: int = HORIZON_STEPS,
) -> tuple[RandomForestRegressor, dict]:
    """Fit the additional forest on observed-only targets over 20 days.

    Each training row's feature values and target come from the same original
    observation series. Rows with a missing latest bin or target are omitted,
    as in the existing forest; other missing history is represented by the
    existing count features and the new slope-validity flags.
    """
    train_end = pd.Timestamp(train_end)
    if not np.isfinite(threshold_w) or threshold_w < 0:
        raise ValueError("threshold_w must be finite and nonnegative")
    if train_days <= 0 or steps <= 0:
        raise ValueError("train_days and steps must be positive")
    train_start = train_end - pd.Timedelta(days=train_days)
    window = observed.loc[(observed.index >= train_start) & (observed.index < train_end)]
    if len(window) != train_days * 96:
        raise ValueError("Forest requires complete calendar-day 15-minute slots")
    features = origin_features(window, threshold_w)
    available_origin = features["latest_w"].notna()
    x_parts: list[pd.DataFrame] = []
    y_parts: list[pd.Series] = []
    counts_by_lead: dict[str, int] = {}
    for lead in range(1, steps + 1):
        target = window.shift(-(lead - 1))
        usable = available_origin & target.notna()
        x_part = features.loc[usable].copy()
        for name, values in target_features(x_part.index, lead).items():
            x_part[name] = values
        x_parts.append(x_part)
        y_parts.append(target.loc[usable])
        counts_by_lead[str(lead)] = int(usable.sum())
    x = pd.concat(x_parts, ignore_index=True)
    y = pd.concat(y_parts, ignore_index=True)
    if len(x) < 1000 or not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError("Insufficient complete observed training examples")
    model = RandomForestRegressor(
        n_estimators=FOREST_TREES,
        min_samples_leaf=FOREST_MIN_SAMPLES_LEAF,
        random_state=FOREST_SEED,
        n_jobs=-1,
    )
    model.fit(x, y)
    return model, {
        "start_inclusive": train_start.isoformat(),
        "end_exclusive": train_end.isoformat(),
        "observed_bins": int(window.notna().sum()),
        "training_rows": len(x),
        "training_rows_by_lead": counts_by_lead,
        "feature_columns": list(x.columns),
        "trees": FOREST_TREES,
        "min_samples_leaf": FOREST_MIN_SAMPLES_LEAF,
        "random_seed": FOREST_SEED,
        "fit_schedule": "once before the evaluation period",
        "peak_activity_threshold_w": float(threshold_w),
        "feature_source": "originally observed bins before each forecast origin",
    }


def predict_augmented_forest(
    model: RandomForestRegressor,
    observed: pd.Series,
    origin: pd.Timestamp,
    threshold_w: float,
    *,
    steps: int = HORIZON_STEPS,
) -> np.ndarray:
    """Predict future 15-minute mean watts using only issue-time inputs."""
    origin = pd.Timestamp(origin)
    if steps <= 0:
        raise ValueError("steps must be positive")
    if origin not in observed.index:
        raise ValueError("Forecast origin is absent from the 15-minute grid")
    # Eight preceding bins suffice for every rolling feature; never inspect
    # the future part of the observed series when issuing a forecast.
    history = observed.loc[origin - 8 * INTERVAL:origin]
    feature = origin_features(history, threshold_w).loc[origin]
    if not np.isfinite(feature.to_numpy(dtype=float)).all():
        raise ValueError("Latest 15-minute source bin must be originally observed")
    x = pd.DataFrame([feature.to_dict()] * steps)
    future = pd.date_range(origin, periods=steps, freq=INTERVAL)
    clock_angle = 2 * np.pi * (future.hour * 60 + future.minute) / 1440
    week_angle = 2 * np.pi * future.dayofweek / 7
    x["target_clock_sin"] = np.sin(clock_angle)
    x["target_clock_cos"] = np.cos(clock_angle)
    x["target_week_sin"] = np.sin(week_angle)
    x["target_week_cos"] = np.cos(week_angle)
    x["lead_step"] = np.arange(1, steps + 1)
    return np.maximum(model.predict(x), 0.0)
