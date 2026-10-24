"""Training-only random-forest correction for the first scenario leads.

The forest uses the same seven flow/calendar features as the existing hybrid
forecast: the four *completed* 15-minute bins in the previous hour, local time
and weekday, and recent draw count/recency. Its two targets are the next two
15-minute mean flows. A single two-output fit is shared by one- and two-step
adaptations, so their comparison changes only the number of corrected leads.

At an issue origin, each selected scenario lead receives a common additive
offset and is floored at zero. The offset is chosen so the weighted scenario
mean equals the forest forecast. This keeps flows physical and retains the
candidate ordering and much of their dispersion where possible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .hybrid import _feature_matrix
from .scenario_methods import ScenarioSet


@dataclass(frozen=True)
class RFShortLeadConfig:
    interval_minutes: int = 15
    draw_threshold_l_per_min: float = 0.75
    forest_trees: int = 100
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.interval_minutes != 15:
            raise ValueError("RF short-lead adaptation currently requires 15-minute intervals")
        if not np.isfinite(self.draw_threshold_l_per_min) or self.draw_threshold_l_per_min < 0:
            raise ValueError("draw_threshold_l_per_min must be finite and nonnegative")
        if isinstance(self.forest_trees, bool) or not isinstance(self.forest_trees, int) or self.forest_trees < 1:
            raise ValueError("forest_trees must be a positive integer")
        if isinstance(self.random_state, bool) or not isinstance(self.random_state, int):
            raise ValueError("random_state must be an integer")


@dataclass(frozen=True)
class RFShortLeadModel:
    """A direct two-output forest fitted wholly before validation begins."""

    estimator: object
    config: RFShortLeadConfig
    training_start: pd.Timestamp
    training_end_exclusive: pd.Timestamp
    training_rows: int

    @property
    def diagnostics(self) -> dict[str, object]:
        return {
            "model": "direct_two_output_random_forest",
            "training_start": self.training_start.isoformat(),
            "training_end_exclusive": self.training_end_exclusive.isoformat(),
            "training_rows": self.training_rows,
            "input_coverage_minutes": 60,
            "target_leads_minutes": [15, 30],
            "features": "same seven completed-hour flow/calendar features as PaperHybrid",
            "forest_trees": self.config.forest_trees,
            "random_state": self.config.random_state,
            "draw_threshold_l_per_min": self.config.draw_threshold_l_per_min,
        }

    def predict(self, history: pd.Series, origin: pd.Timestamp) -> np.ndarray | None:
        """Predict two leads using only bins that end no later than ``origin``.

        Returns ``None`` if the immediately preceding hour has a missing bin.
        Values supplied at or after the origin are never included as features.
        """
        _validate_index(history, "history")
        origin = pd.Timestamp(origin)
        if origin.tz is None:
            raise ValueError("origin must be timezone-aware")
        # A pandas Timestamp's pytz object carries its date-specific offset,
        # whereas a DatetimeIndex holds the zone object; compare zone names.
        if str(history.index.tz) != str(self.training_start.tz):
            raise ValueError("history timezone must match the fitted training timezone")
        origin = origin.tz_convert(history.index.tz)
        if origin < self.training_end_exclusive:
            raise ValueError("RF forecasts must start at or after the training cutoff")
        interval = pd.Timedelta(minutes=self.config.interval_minutes)
        past_index = pd.date_range(end=origin - interval, periods=4, freq=interval)
        past_hour = history.reindex(past_index).to_numpy(dtype=float)
        if not np.isfinite(past_hour).all():
            return None
        if (past_hour < 0).any():
            raise ValueError("Completed water flow must be nonnegative")
        features = _feature_matrix(
            past_hour[None, :], pd.DatetimeIndex([origin]),
            self.config.draw_threshold_l_per_min,
        )
        forecast = np.asarray(self.estimator.predict(features), dtype=float).reshape(-1)
        if forecast.shape != (2,) or not np.isfinite(forecast).all():
            raise ValueError("Random forest must predict two finite future flow values")
        return np.maximum(forecast, 0.0)


def _validate_index(series: pd.Series, label: str) -> None:
    if not isinstance(series, pd.Series) or not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError(f"{label} must be a time-indexed Series")
    if series.index.tz is None or series.index.has_duplicates or not series.index.is_monotonic_increasing:
        raise ValueError(f"{label} needs unique, increasing timezone-aware timestamps")


def fit_rf_shortlead(
    training: pd.Series, config: RFShortLeadConfig | None = None
) -> RFShortLeadModel:
    """Fit once on training intervals only; both future targets end by its cutoff."""
    config = config or RFShortLeadConfig()
    _validate_index(training, "training")
    interval = pd.Timedelta(minutes=config.interval_minutes)
    if len(training) < 6 or not training.index.equals(pd.date_range(
        training.index[0], periods=len(training), freq=interval
    )):
        raise ValueError("training needs at least six regularly spaced 15-minute bins")
    values = training.to_numpy(dtype=float)
    if np.isinf(values).any() or (values[np.isfinite(values)] < 0).any():
        raise ValueError("training flow must be nonnegative and finite or missing")
    # Four completed bins precede each issue time. The two labels are the two
    # following bins, both strictly within the supplied training series.
    rows = len(values) - 4 - 2 + 1
    past = np.lib.stride_tricks.sliding_window_view(values, 4)[:rows]
    targets = np.lib.stride_tricks.sliding_window_view(values, 2)[4:4 + rows]
    valid = np.isfinite(past).all(axis=1) & np.isfinite(targets).all(axis=1)
    if not valid.any():
        raise ValueError("No complete training row has a prior hour and two future targets")
    origins = training.index[4:4 + rows]
    features = _feature_matrix(
        past[valid], origins[valid], config.draw_threshold_l_per_min
    )
    try:
        from sklearn.ensemble import RandomForestRegressor
    except ImportError as error:
        raise ImportError(
            "RF short-lead adaptation requires scikit-learn; install "
            "requirements-water-forecasting.txt"
        ) from error
    estimator = RandomForestRegressor(
        n_estimators=config.forest_trees,
        random_state=config.random_state,
        n_jobs=1,
    )
    estimator.fit(features, targets[valid])
    return RFShortLeadModel(
        estimator=estimator,
        config=config,
        training_start=training.index[0],
        training_end_exclusive=training.index[-1] + interval,
        training_rows=int(valid.sum()),
    )


def _project_nonnegative_mean(
    values: np.ndarray, weights: np.ndarray, target: float
) -> tuple[np.ndarray, float]:
    """Find a common offset with weighted mean ``target`` after zero flooring.

    For a downward correction, the monotone mean function is bracketed by
    ``-max(values)`` (all zero) and zero (the original mean). A bounded
    bisection handles arbitrary zero-heavy empirical scenario distributions.
    """
    if target == 0:
        return np.zeros_like(values), -float(np.max(values))
    original_mean = float(weights @ values)
    if target >= original_mean:
        offset = (target - original_mean) / float(weights.sum())
        return values + offset, offset
    lower, upper = -float(np.max(values)), 0.0
    for _ in range(80):
        midpoint = lower + (upper - lower) / 2
        if float(weights @ np.maximum(values + midpoint, 0.0)) < target:
            lower = midpoint
        else:
            upper = midpoint
    offset = lower + (upper - lower) / 2
    return np.maximum(values + offset, 0.0), offset


def apply_rf_shortlead(
    scenarios: ScenarioSet,
    history: pd.Series,
    model: RFShortLeadModel,
    *,
    steps: int,
) -> ScenarioSet:
    """Causally project the first one or two scenario means onto the RF values."""
    if isinstance(steps, bool) or not isinstance(steps, int) or steps not in {1, 2}:
        raise ValueError("steps must be one or two 15-minute intervals")
    values = np.asarray(scenarios.values, dtype=float)
    weights = np.asarray(scenarios.weights, dtype=float)
    if values.ndim != 2 or values.shape[1] < steps or len(values) != len(weights):
        raise ValueError("scenario paths and weights must contain the selected leads")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("scenario flows must be nonnegative and finite")
    if not np.isfinite(weights).all() or (weights <= 0).any() or not np.isclose(weights.sum(), 1):
        raise ValueError("scenario weights must be positive and normalized")
    if scenarios.diagnostics.get("rf_shortlead_steps_applied", 0):
        raise ValueError("Apply RF correction to a base scenario set, not an already corrected set")
    if scenarios.diagnostics.get("persistence_steps_applied", 0):
        raise ValueError("Apply RF correction instead of persistence, not after it")
    origin = pd.Timestamp(scenarios.diagnostics["origin"])
    predicted = model.predict(history, origin)
    diagnostics = dict(scenarios.diagnostics)
    diagnostics.update({
        "rf_shortlead_steps_requested": steps,
        "rf_shortlead_steps_applied": steps if predicted is not None else 0,
        "rf_shortlead_skip_reason": "missing_complete_prior_hour" if predicted is None else None,
        "rf_training_end_exclusive": model.training_end_exclusive.isoformat(),
        "rf_training_rows": model.training_rows,
        "rf_input_end_exclusive": origin.isoformat(),
        "rf_predicted_l_per_min": None if predicted is None else predicted[:steps].tolist(),
    })
    if predicted is None:
        return ScenarioSet(values, weights, scenarios.candidate_ids, diagnostics)
    base_mean = weights @ values[:, :steps]
    adapted = values.copy()
    offsets = []
    for lead in range(steps):
        adapted[:, lead], offset = _project_nonnegative_mean(
            values[:, lead], weights, float(predicted[lead])
        )
        offsets.append(offset)
    diagnostics["rf_base_weighted_mean_l_per_min"] = base_mean.tolist()
    diagnostics["rf_adapted_weighted_mean_l_per_min"] = (weights @ adapted[:, :steps]).tolist()
    diagnostics["rf_projection_offsets_l_per_min"] = offsets
    diagnostics["rf_projection_zero_floor_active"] = [
        bool((adapted[:, lead] == 0).any() and offsets[lead] < 0)
        for lead in range(steps)
    ]
    diagnostics["rf_target_mean_error_l_per_min"] = (
        weights @ adapted[:, :steps] - predicted[:steps]
    ).tolist()
    return ScenarioSet(adapted, weights, scenarios.candidate_ids, diagnostics)
