"""The paper's random-forest / Prophet / historical-persistence hybrid.

Section 3.3 uses direct random-forest forecasts for short leads, Prophet for
medium leads, and a previous-period profile for long leads. Section 4.2.2
(page 18) supplies the inclusive 20- and 500-minute routing boundaries.
This flow-only adaptation omits the tank-temperature feature because the
occupant CSV contains no temperature measurements. Its draw threshold is
configurable; the paper does not prescribe a numerical threshold.
If the model timestep exceeds the short-lead cutoff, no interval ends in
the RF range, so only Prophet and persistence are fitted and routed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HybridConfig:
    """Physical routing cutoffs, independent of the forecast horizon and grid."""

    short_horizon_minutes: int = 20
    medium_horizon_minutes: int = 500
    draw_threshold_l_per_min: float = 0.75
    persistence_strategy: str = "previous_day"
    random_state: int = 42
    forest_trees: int = 100

    def __post_init__(self) -> None:
        for name in ("short_horizon_minutes", "medium_horizon_minutes", "forest_trees"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.short_horizon_minutes >= self.medium_horizon_minutes:
            raise ValueError("Routing cutoffs must satisfy 0 < short < medium")
        if not np.isfinite(self.draw_threshold_l_per_min) or self.draw_threshold_l_per_min < 0:
            raise ValueError("Draw threshold must be finite and nonnegative")
        if self.persistence_strategy not in {"previous_day", "persistence", "previous_week"}:
            raise ValueError("Persistence strategy must be previous_day, persistence, or previous_week")


def _feature_matrix(
    past_hour: np.ndarray, origins: pd.DatetimeIndex, threshold: float
) -> np.ndarray:
    """Seven features from completed intervals and known local calendar time."""
    hours = np.asarray(origins.hour + origins.minute / 60 + origins.second / 3600)
    weekdays = np.asarray(origins.dayofweek)
    draws = past_hour >= threshold
    recency = np.where(
        draws.any(axis=1), draws[:, ::-1].argmax(axis=1) + 1, past_hour.shape[1] + 1
    )
    return np.column_stack([
        past_hour[:, -1],
        np.sin(2 * np.pi * hours / 24),
        np.cos(2 * np.pi * hours / 24),
        np.sin(2 * np.pi * weekdays / 7),
        np.cos(2 * np.pi * weekdays / 7),
        draws.sum(axis=1),
        recency,
    ])


class PaperHybrid:
    """Route each interval by its ending lead; never blend component outputs.

    Fit only on observations supplied to ``fit``. ``predict`` may use newly
    completed flow for RF features and historical persistence, but does not
    retrain either learned model. The comparison runner controls refitting.
    Any positive number of future intervals is supported, including a 12-hour
    horizon, without changing the physical component cutoffs.
    """

    def __init__(self, config: HybridConfig | None = None) -> None:
        # Local imports keep the strategy registry free of circular imports.
        from .strategies import DailyProfile, Persistence, PreviousDay, PreviousWeek

        self.config = config if config is not None else HybridConfig()
        self._profile = DailyProfile()
        self._persistence = {
            "previous_day": PreviousDay,
            "persistence": Persistence,
            "previous_week": PreviousWeek,
        }[self.config.persistence_strategy]()
        self._forest = None
        self._prophet = None
        self._prophet_cache: dict[pd.Timestamp, float] = {}
        self._interval: pd.Timedelta | None = None
        self._interval_minutes: int | None = None
        self._hour_steps: int | None = None
        self._short_steps: int | None = None
        self._timezone = None
        self.diagnostics: dict = {}

    def fit(self, training: pd.Series) -> None:
        """Fit direct short-lead RF targets and real Prophet on training only.

        RF rows require a complete prior hour and complete short-lead targets.
        Prophet uses all finite training points; timezone stripping preserves
        local daily/weekly cycles, and repeated DST clock times are averaged.
        """
        if not isinstance(training.index, pd.DatetimeIndex) or training.index.tz is None:
            raise ValueError("Training must have a timezone-aware DatetimeIndex")
        index = training.index
        if len(index) < 2 or index.has_duplicates or not index.is_monotonic_increasing:
            raise ValueError("Training timestamps must be unique, increasing, and regularly spaced")
        differences = np.diff(index.asi8)
        interval = pd.Timedelta(int(differences[0]), unit="ns")
        minute = pd.Timedelta(minutes=1)
        if interval <= pd.Timedelta(0) or not np.all(differences == differences[0]):
            raise ValueError("Training timestamps must be regularly spaced")
        if interval % minute:
            raise ValueError("Model interval must be a positive whole number of minutes")
        interval_minutes = int(interval // minute)
        if 60 % interval_minutes and (interval_minutes % 60 or 1440 % interval_minutes):
            raise ValueError("Model interval must divide 60 minutes or be whole hours dividing a day")
        values = training.to_numpy(dtype=float)
        if np.isinf(values).any() or (values < 0).any():
            raise ValueError("Training flow must be nonnegative and finite, or missing")

        hour_steps = max(1, int(np.ceil(60 / interval_minutes)))
        short_steps = self.config.short_horizon_minutes // interval_minutes
        try:
            from prophet import Prophet
        except ImportError as error:
            raise ImportError(
                "PaperHybrid requires prophet; install requirements-water-forecasting.txt"
            ) from error
        forest = None
        rf_training_rows = 0
        if short_steps:
            row_count = len(values) - hour_steps - short_steps + 1
            if row_count <= 0:
                raise ValueError("Training is too short for a complete past hour and short-lead targets")
            past_hour = np.lib.stride_tricks.sliding_window_view(values, hour_steps)[:row_count]
            targets = np.lib.stride_tricks.sliding_window_view(values, short_steps)[
                hour_steps:hour_steps + row_count
            ]
            valid = np.isfinite(past_hour).all(axis=1) & np.isfinite(targets).all(axis=1)
            if not valid.any():
                raise ValueError("No complete RF training rows: each needs a past hour and short-lead targets")
            origins = index[hour_steps:hour_steps + row_count]
            features = _feature_matrix(past_hour[valid], origins[valid], self.config.draw_threshold_l_per_min)
            try:
                from sklearn.ensemble import RandomForestRegressor
            except ImportError as error:
                raise ImportError(
                    "Short-lead PaperHybrid forecasts require scikit-learn; "
                    "install requirements-water-forecasting.txt"
                ) from error
            forest = RandomForestRegressor(
                n_estimators=self.config.forest_trees,
                random_state=self.config.random_state,
                n_jobs=1,
            )
            target_values = targets[valid]
            forest.fit(features, target_values[:, 0] if short_steps == 1 else target_values)
            rf_training_rows = int(valid.sum())

        finite = np.isfinite(values)
        prophet_training = pd.DataFrame({
            "ds": index[finite].tz_localize(None), "y": values[finite]
        }).groupby("ds", as_index=False, sort=True)["y"].mean()
        prophet = Prophet(
            growth="linear",
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
            uncertainty_samples=0,
        )
        prophet.fit(prophet_training)
        self._profile.fit(training)
        self._persistence.fit(training)
        self._forest = forest
        self._prophet = prophet
        self._prophet_cache = {}
        self._interval = interval
        self._interval_minutes = interval_minutes
        self._hour_steps = hour_steps
        self._short_steps = short_steps
        self._timezone = index.tz
        self.diagnostics = {
            "model_interval_minutes": interval_minutes,
            "short_horizon_minutes": self.config.short_horizon_minutes,
            "medium_horizon_minutes": self.config.medium_horizon_minutes,
            "random_forest_output_bins": short_steps,
            "random_forest_training_rows": rf_training_rows,
            "random_forest_fitted": forest is not None,
            "routing_policy": "Route each interval using its ending lead and unchanged physical cutoffs",
            "coarse_interval_policy": (
                "No RF interval ends within the short cutoff; fit Prophet and persistence only"
                if not short_steps else
                "Fit direct RF targets for intervals ending within the short cutoff"
            ),
            "random_forest_input_coverage_minutes": hour_steps * interval_minutes if short_steps else None,
            "prophet_training_rows": len(prophet_training),
        }

    def _validate_future(self, future_index: pd.DatetimeIndex) -> None:
        if self._prophet is None or self._interval is None:
            raise RuntimeError("Call fit before predict")
        if not isinstance(future_index, pd.DatetimeIndex) or future_index.tz != self._timezone:
            raise ValueError("Future timestamps must use the fitted local timezone")
        if future_index.has_duplicates or not future_index.is_monotonic_increasing:
            raise ValueError("Future timestamps must be unique and increasing")
        if len(future_index) and not future_index.equals(pd.date_range(
            future_index[0], periods=len(future_index), freq=self._interval
        )):
            raise ValueError("Future timestamps must follow the fitted model interval")

    def component_names(self, future_index: pd.DatetimeIndex) -> np.ndarray:
        """Name each routed component using inclusive physical lead cutoffs."""
        self._validate_future(future_index)
        leads = (np.arange(len(future_index)) + 1) * self._interval_minutes
        return np.where(
            leads <= self.config.short_horizon_minutes,
            "random_forest",
            np.where(leads <= self.config.medium_horizon_minutes, "prophet", self.config.persistence_strategy),
        )

    def _features(self, history: pd.Series, origin: pd.Timestamp) -> np.ndarray | None:
        """Extract one origin's seven features, or None for an incomplete hour.

        The interval at ``origin`` is unobserved. The latest accepted interval
        starts at ``origin - interval`` and ends at the forecast origin.
        """
        if self._interval is None:
            raise RuntimeError("Call fit before extracting features")
        past_index = pd.date_range(
            end=origin - self._interval, periods=self._hour_steps, freq=self._interval
        )
        past_hour = history.reindex(past_index).to_numpy(dtype=float)
        if not np.isfinite(past_hour).all():
            return None
        return _feature_matrix(
            past_hour[None, :], pd.DatetimeIndex([origin]), self.config.draw_threshold_l_per_min
        )[0]

    def _prophet_predictions(self, future_index: pd.DatetimeIndex) -> np.ndarray:
        """Cache local-clock point forecasts until the next explicit fit."""
        local = future_index.tz_localize(None)
        missing = [timestamp for timestamp in local.unique() if timestamp not in self._prophet_cache]
        if missing:
            forecast = self._prophet.predict(pd.DataFrame({"ds": missing}))
            self._prophet_cache.update({
                pd.Timestamp(timestamp): float(value)
                for timestamp, value in zip(forecast["ds"], forecast["yhat"])
            })
        return np.asarray([self._prophet_cache[timestamp] for timestamp in local], dtype=float)

    def predict(self, history: pd.Series, future_index: pd.DatetimeIndex) -> np.ndarray:
        """Forecast arbitrary-length futures from strictly completed history.

        Beyond one historical period, persistence repeats the earlier final
        hybrid prediction at the analogous local time within this forecast.
        """
        components = self.component_names(future_index)
        if future_index.empty:
            return np.empty(0, dtype=float)
        if not isinstance(history.index, pd.DatetimeIndex) or history.index.tz != self._timezone:
            raise ValueError("History must use the fitted local timezone")
        if history.index.has_duplicates or not history.index.is_monotonic_increasing:
            raise ValueError("History timestamps must be unique and increasing")
        completed = history.loc[history.index < future_index[0]]
        predicted = np.empty(len(future_index), dtype=float)
        short = components == "random_forest"
        if short.any():
            features = self._features(completed, future_index[0])
            if features is None:
                predicted[short] = self._profile.predict(completed, future_index[short])
            else:
                short_forecast = np.asarray(self._forest.predict(features[None, :])).reshape(-1)
                predicted[short] = short_forecast[:int(short.sum())]
        medium = components == "prophet"
        if medium.any():
            predicted[medium] = self._prophet_predictions(future_index[medium])
        long = ~(short | medium)
        if long.any():
            # The baseline handles observed analogs and its fitted fallback.
            predicted[long] = self._persistence.predict(completed, future_index)[long]
            if self.config.persistence_strategy in {"previous_day", "previous_week"}:
                days = 1 if self.config.persistence_strategy == "previous_day" else 7
                offset = pd.DateOffset(days=days)
                earlier_forecasts: dict[pd.Timestamp, float] = {}
                for position, local_time in enumerate(future_index.tz_localize(None)):
                    analog = local_time - offset
                    if long[position] and analog in earlier_forecasts:
                        predicted[position] = earlier_forecasts[analog]
                    earlier_forecasts[local_time] = max(float(predicted[position]), 0.0)
        if not np.isfinite(predicted).all():
            raise ValueError("Hybrid components returned nonfinite forecasts")
        return np.maximum(predicted, 0.0)
