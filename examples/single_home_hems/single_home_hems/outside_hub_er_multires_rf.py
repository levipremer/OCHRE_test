"""Deploy the four-week Outside-Hub-plus-resistance multiresolution RF.

The model is fitted with the exact observed-target rows and feature order of
the offline forecast study. At issue time it sees only completed total-home
and corrected Home Hub aggregate meters. Electric resistance is part of their
difference; no resistance circuit reading is passed to the predictor.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .runtime_threads import apply_rf_threads

from examples.single_home_hems.forecasting.evaluate_outside_hub import SIMULATION_START
from examples.single_home_hems.forecasting.evaluate_simulated_home_hub import HORIZON_STEPS, WEEK
from examples.single_home_hems.forecasting.outside_hub_er_multires_load import load_feature_matrix
from .home_hub_observation import HUB_POWER_KEY, TOTAL_POWER_KEY
from .prior_day_load_forecast import (
    EXTENDED_STEPS, RF_STEPS, clock_slot_forecast, prior_complete_local_day,
)


METHOD = "rf_multires_weather_persistence"
INTERVAL = pd.Timedelta(minutes=15)
MINUTE = pd.Timedelta(minutes=1)
HOUR = pd.Timedelta(hours=1)
TRAIN_START = SIMULATION_START + WEEK
TRAIN_END = SIMULATION_START + 5 * WEEK
PREDICTIVE_INPUTS = Path(__file__).resolve().parents[1] / "inputs/predictive"
STUDY_METADATA = PREDICTIVE_INPUTS / "models/outside_hub_er_study_metadata.json"
PRETEST_DIR = PREDICTIVE_INPUTS / "seeds"
DEFAULT_ARTIFACT = PREDICTIVE_INPUTS / "models/outside_hub_er_multires_rf.joblib"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _local_naive(index: pd.DatetimeIndex, timezone: Any) -> pd.DatetimeIndex:
    if index.tz is None:
        return index
    return index.tz_convert(timezone).tz_localize(None)


def fit_study_model(path: str | Path = DEFAULT_ARTIFACT) -> Path:
    """Save the study's selected four-week model with its checked feature schema."""
    from examples.single_home_hems.forecasting.evaluate_outside_hub import supervised_rows
    from examples.single_home_hems.forecasting.evaluate_outside_hub_with_er import load_composite_target
    from examples.single_home_hems.forecasting.evaluate_outside_hub_er_multires_rf import (
        HORIZON_STEPS, OLD_LAGS, WEATHER_MINUTE,
    )
    from examples.single_home_hems.forecasting.outside_hub_candidate_models import make_candidate
    from examples.single_home_hems.forecasting.outside_hub_er_multires_load import (
        load_feature_matrix, load_minute_observed,
    )
    from examples.single_home_hems.forecasting.outside_hub_er_weather_features import weather_feature_matrices

    path = Path(path).expanduser().resolve()
    study = json.loads(STUDY_METADATA.read_text(encoding="utf-8"))
    chosen = study["selected_validation_params"][f"4w:{METHOD}"]
    columns = tuple(study["feature_columns"][METHOD])
    if chosen["min_samples_leaf"] != 5 or len(columns) != 205:
        raise ValueError("Saved four-week multiresolution RF selection changed")
    truth, observed, target_audit = load_composite_target()
    rows, observed_labels = supervised_rows(observed, truth)
    fit = rows.mask_fit(TRAIN_START, TRAIN_END) & np.isfinite(observed_labels)
    if not fit.any():
        raise ValueError("Selected four-week RF has no originally observed training labels")
    minute_observed, minute_audit = load_minute_observed(truth, observed)
    origins = pd.DatetimeIndex(rows.origin[fit])
    targets = pd.DatetimeIndex(rows.target[fit])
    multires = load_feature_matrix(minute_observed, origins.unique())
    repeated = multires.loc[origins].reset_index(drop=True)
    retained = rows.x.loc[fit].drop(columns=list(OLD_LAGS)).reset_index(drop=True)
    weather = weather_feature_matrices(origins, targets, WEATHER_MINUTE)[
        "past_weather_persistence"
    ]
    features = pd.concat((retained, repeated, weather), axis=1)
    if tuple(features.columns) != columns:
        raise ValueError("Training feature order differs from approved study metadata")
    x = np.ascontiguousarray(features.to_numpy(dtype=np.float32))
    y = np.asarray(observed_labels[fit], dtype=float)
    if x.shape != (len(y), len(columns)) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Selected RF fit data are incomplete")
    model = make_candidate("rf_history", {"min_samples_leaf": 5})
    model.fit(x, y)
    metadata = {
        "method_label": METHOD,
        "target": "15-minute mean total-home minus corrected Home Hub aggregate, including HVAC electric resistance",
        "runtime_input_rule": "completed total-home aggregate minus completed corrected Home Hub aggregate; no resistance submeter",
        "training_start_inclusive": TRAIN_START.isoformat(),
        "training_end_exclusive": TRAIN_END.isoformat(),
        "training_power_rule": "originally observed aggregate target bins only",
        "fit_pairs": len(y),
        "fit_origin_count": len(origins.unique()),
        "forecast_interval_minutes": 15,
        "forecast_horizon_steps": HORIZON_STEPS,
        "weather_rule": "24 completed prior hourly temperature and relative-humidity means; latest complete hourly mean persisted across every future lead",
        "load_feature_rule": "15 completed minute readings; 7 complete quarter-hour blocks; 22 complete hourly blocks; per-feature availability flags",
        "rf_spec": study["rf_spec"],
        "min_samples_leaf": 5,
        "validation_mae_kw": chosen["cv_mae_kw"],
        "feature_columns": list(columns),
        "study_metadata_sha256": _sha256(STUDY_METADATA),
        "training_weather_sha256": _sha256(WEATHER_MINUTE),
        "target_audit": target_audit,
        "minute_audit": minute_audit,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    joblib.dump({"model": model, "feature_columns": columns, "metadata": metadata}, temporary)
    temporary.replace(path)
    return path


def _read_pretest_seed(pretest_dir: Path, start: pd.Timestamp, timezone: Any) -> pd.Series:
    """Read the checked pretest aggregate, retaining the legacy meter reader."""
    prepared = pretest_dir / "outside_hub_pretest_minute.csv"
    if prepared.is_file():
        frame = pd.read_csv(
            prepared, usecols=["completed_at", "outside_hub_with_resistance_kw"],
            parse_dates=["completed_at"], float_precision="round_trip",
        )
        aggregate = frame.set_index("completed_at")["outside_hub_with_resistance_kw"]
        aggregate.index = pd.DatetimeIndex(aggregate.index, name="completed_at")
    else:
        meter = pd.read_csv(
            pretest_dir / "home_hub_meter_minute.csv",
            usecols=["timestamp", "home_hub_power_kw"], parse_dates=["timestamp"],
        ).set_index("timestamp")["home_hub_power_kw"]
        total = pd.read_csv(
            pretest_dir / "home_01_standardized.csv",
            usecols=["timestamp", "total_electric_power_kw"], parse_dates=["timestamp"],
        ).set_index("timestamp")["total_electric_power_kw"]
        er = pd.read_csv(
            pretest_dir / "raw/home_01.csv",
            usecols=["Time", "HVAC Heating ER Power (kW)"], parse_dates=["Time"],
        ).set_index("Time")["HVAC Heating ER Power (kW)"]
        if not meter.index.equals(total.index):
            raise ValueError("Pretest total-home and Home Hub meters are not aligned")
        er = er.reindex(meter.index)
        if er.isna().any():
            raise ValueError("Pretest ER wiring correction lacks a minute")
        corrected_hub = meter - er
        aggregate = (total - corrected_hub).astype(float)
        end_stamps = _local_naive(pd.DatetimeIndex(aggregate.index), timezone) + MINUTE
        aggregate.index = pd.DatetimeIndex(end_stamps, name="completed_at")
    if aggregate.index.has_duplicates or not aggregate.index.is_monotonic_increasing:
        raise ValueError("Pretest aggregate timestamps are not unique and ordered")
    if not np.isfinite(aggregate.to_numpy()).all() or aggregate.lt(-1e-7).any():
        raise ValueError("Pretest corrected aggregate Outside-Hub history is invalid")
    wall = start.tz_localize(None) if start.tzinfo is not None else start
    seed = aggregate.loc[(aggregate.index > wall - 24 * HOUR) & (aggregate.index <= wall)]
    if len(seed) != 24 * 60 or seed.index[-1] != wall:
        raise ValueError("Pretest aggregate seed lacks the complete day before control")
    return seed


def _read_runtime_weather(
    path: Path, start: pd.Timestamp, end: pd.Timestamp, timezone: Any,
) -> pd.DataFrame:
    weather = pd.read_csv(
        path, usecols=["Time", "temp_air", "relative_humidity"],
    )
    index = pd.DatetimeIndex(pd.to_datetime(weather["Time"], utc=True))
    index = _local_naive(index, timezone)
    weather = pd.DataFrame({
        "ambient_temperature_c": pd.to_numeric(weather["temp_air"], errors="raise").to_numpy(float),
        "ambient_relative_humidity": pd.to_numeric(weather["relative_humidity"], errors="raise").to_numpy(float),
    }, index=index)
    left = start.tz_localize(None) if start.tzinfo is not None else start
    right = end.tz_localize(None) if end.tzinfo is not None else end
    weather = weather.loc[(weather.index >= left - 24 * HOUR) & (weather.index < right)]
    expected = pd.date_range(left - 24 * HOUR, right, freq="min", inclusive="left")
    if not weather.index.equals(expected) or not np.isfinite(weather.to_numpy()).all():
        raise ValueError("Runtime weather lacks complete past-observed minute coverage")
    humidity = weather["ambient_relative_humidity"]
    if humidity.lt(0).any() or humidity.gt(1).any():
        raise ValueError("Runtime relative humidity must be a fraction in [0, 1]")
    return weather


class MultiresWeatherOutsideHubRF:
    """Issue causal 32-step Outside-Hub forecasts from two completed meters."""

    requires_total_home_meter = True
    method_label = METHOD

    def __init__(
        self, model: Any, feature_columns: tuple[str, ...], metadata: dict[str, Any],
        seed: pd.Series, weather: pd.DataFrame, timezone: Any,
        tail_policy: str = "none",
    ) -> None:
        if tail_policy not in {"none", "prior_complete_local_day"}:
            raise ValueError("Unknown Outside-Hub forecast tail policy")
        self.model = model
        self.feature_columns = feature_columns
        self.metadata = metadata
        self.seed = seed
        self.weather = weather
        self.timezone = timezone
        self.tail_policy = tail_policy
        if tail_policy == "prior_complete_local_day":
            self.method_label = f"{METHOD}__prior_complete_local_day_after_8h"
            self.metadata = dict(metadata, method_label=self.method_label,
                                 tail_policy=tail_policy, tail_first_step=RF_STEPS)
        self.past_hour_weather = weather.rolling(60, min_periods=60).mean().shift(1)
        self.history: list[dict[str, Any]] = []

    @classmethod
    def from_artifact(
        cls, artifact: str | Path, *, pretest_dir: str | Path = PRETEST_DIR,
        weather_file: str | Path, simulation_start: pd.Timestamp,
        simulation_end: pd.Timestamp, tail_policy: str = "none",
    ) -> "MultiresWeatherOutsideHubRF":
        path = Path(artifact).expanduser().resolve()
        saved = joblib.load(path)
        if not isinstance(saved, dict) or set(saved) != {"model", "feature_columns", "metadata"}:
            raise ValueError("Multiresolution RF artifact has an unexpected schema")
        apply_rf_threads(saved["model"])
        study = json.loads(STUDY_METADATA.read_text(encoding="utf-8"))
        columns = tuple(study["feature_columns"][METHOD])
        if tuple(saved["feature_columns"]) != columns or len(columns) != 205:
            raise ValueError("Runtime RF feature schema differs from approved study")
        metadata = dict(saved["metadata"])
        if (metadata.get("method_label") != METHOD
                or metadata.get("study_metadata_sha256") != _sha256(STUDY_METADATA)
                or metadata.get("training_start_inclusive") != TRAIN_START.isoformat()
                or metadata.get("training_end_exclusive") != TRAIN_END.isoformat()
                or metadata.get("min_samples_leaf") != 5):
            raise ValueError("Multiresolution RF artifact differs from selected four-week study")
        timezone = simulation_start.tz
        seed = _read_pretest_seed(Path(pretest_dir), simulation_start, timezone)
        weather = _read_runtime_weather(
            Path(weather_file), simulation_start, simulation_end, timezone,
        )
        metadata.update({
            "artifact_path": str(path), "artifact_sha256": _sha256(path),
            "pretest_seed_path": str(Path(pretest_dir).resolve()),
            "runtime_weather_path": str(Path(weather_file).resolve()),
            "runtime_weather_sha256": _sha256(Path(weather_file)),
            "seed_minutes": len(seed),
        })
        return cls(saved["model"], columns, metadata, seed, weather, timezone,
                   tail_policy=tail_policy)

    def _live_minute_history(
        self, timestamp: pd.Timestamp, completed_history: pd.DataFrame,
    ) -> pd.Series:
        needed = {HUB_POWER_KEY, TOTAL_POWER_KEY}
        if not needed.issubset(completed_history):
            raise ValueError("Outside-Hub RF requires completed total-home and corrected Hub meters")
        if not isinstance(completed_history.index, pd.DatetimeIndex):
            raise TypeError("Outside-Hub RF completed meter times must be timestamps")
        if completed_history.index.has_duplicates or not completed_history.index.is_monotonic_increasing:
            raise ValueError("Outside-Hub RF completed meter times must be unique and increasing")
        available = completed_history.loc[completed_history.index <= timestamp]
        if available.empty or available.index[-1] != timestamp:
            raise ValueError("Outside-Hub RF lacks the latest completed aggregate meter reading")
        minute = (available[TOTAL_POWER_KEY].astype(float)
                  - available[HUB_POWER_KEY].astype(float))
        minute.index = _local_naive(pd.DatetimeIndex(minute.index), self.timezone)
        values = minute.to_numpy(dtype=float)
        if (np.isinf(values).any() or np.any(values[np.isfinite(values)] < -1e-7)
                or (self.tail_policy == "none" and not np.isfinite(values).all())):
            raise ValueError("Completed Outside-Hub aggregate meter history is invalid")
        minute.name = "outside_hub_with_resistance_kw"
        return minute

    def _features(self, wall: pd.Timestamp, minute: pd.Series) -> pd.DataFrame:
        expected = pd.date_range(wall - 24 * HOUR + MINUTE, wall, freq="min", name="completed_at")
        recent = minute.reindex(expected)
        if np.isinf(recent.to_numpy(float)).any():
            raise ValueError("Outside-Hub RF aggregate meter history contains infinite values")
        multires = load_feature_matrix(recent, pd.DatetimeIndex([wall])).iloc[0]
        grouped = recent.resample("15min", closed="right", label="right")
        means, counts = grouped.mean(), grouped.count()
        means = means.where(counts.eq(15))
        means.index -= INTERVAL
        bins = means.reindex(pd.date_range(wall - 24 * HOUR, wall - INTERVAL, freq=INTERVAL))
        if len(bins) != 96:
            raise ValueError("Outside-Hub RF lacks trailing 15-minute aggregate bins")
        base: dict[str, float | int] = {}
        for steps in (4, 8, 96):
            values = bins.iloc[-steps:]
            base[f"load_mean_prev_{steps}_kw"] = float(values.mean()) if values.notna().any() else 0.0
            base[f"load_max_prev_{steps}_kw"] = float(values.max()) if values.notna().any() else 0.0
            base[f"load_count_prev_{steps}"] = int(values.notna().sum())
        slope = float(bins.iloc[-1] - bins.iloc[-5])
        base["load_slope_1h_kw"] = slope if np.isfinite(slope) else 0.0
        base["load_slope_1h_available"] = int(np.isfinite(slope))
        targets = pd.date_range(wall, periods=HORIZON_STEPS, freq=INTERVAL)
        clock = 2 * np.pi * (targets.hour * 60 + targets.minute) / 1440
        week = 2 * np.pi * targets.dayofweek / 7
        values: dict[str, np.ndarray] = {
            key: np.repeat(value, HORIZON_STEPS) for key, value in base.items()
        }
        values["lead_step"] = np.arange(1, HORIZON_STEPS + 1)
        values["target_clock_sin"] = np.sin(clock)
        values["target_clock_cos"] = np.cos(clock)
        values["target_week_sin"] = np.sin(week)
        values["target_week_cos"] = np.cos(week)
        values["target_is_weekend"] = (targets.dayofweek >= 5).astype(int)
        for name, value in multires.items():
            values[name] = np.repeat(value, HORIZON_STEPS)
        future: dict[str, float | int] = {}
        for prefix, unit, source in (
            ("ambient_temperature", "c", "ambient_temperature_c"),
            ("ambient_relative_humidity", "", "ambient_relative_humidity"),
        ):
            for hour in range(1, 25):
                lookback = wall - (hour - 1) * HOUR
                value = self.past_hour_weather[source].get(lookback, np.nan)
                present = bool(np.isfinite(value))
                stem = f"{prefix}_prev_{hour}h"
                values[f"{stem}_{unit}" if unit else stem] = np.repeat(
                    float(value) if present else 0.0, HORIZON_STEPS,
                )
                values[f"{stem}_available"] = np.repeat(int(present), HORIZON_STEPS)
                if hour == 1:
                    future_stem = f"future_{prefix}"
                    future_value_name = f"{future_stem}_{unit}" if unit else future_stem
                    future[future_value_name] = float(value) if present else 0.0
                    future[f"{future_stem}_available"] = int(present)
        for name, value in future.items():
            values[name] = np.repeat(value, HORIZON_STEPS)
        features = pd.DataFrame(values)
        if tuple(features.columns) != self.feature_columns:
            raise ValueError("Runtime Outside-Hub features differ from approved study schema")
        values = features.to_numpy(dtype=np.float32)
        if values.shape != (HORIZON_STEPS, 205) or not np.isfinite(values).all():
            raise ValueError("Runtime Outside-Hub RF feature matrix is incomplete")
        return features

    def forecast_from_completed(
        self, timestamp: pd.Timestamp, completed_history: pd.DataFrame,
        *, steps: int = HORIZON_STEPS,
    ) -> pd.Series:
        timestamp = pd.Timestamp(timestamp)
        allowed = {HORIZON_STEPS}
        if self.tail_policy == "prior_complete_local_day":
            allowed.add(EXTENDED_STEPS)
        if timestamp != timestamp.floor(INTERVAL) or steps not in allowed:
            raise ValueError("Multiresolution RF requires an aligned supported eight- or 20-hour issue")
        wall = timestamp.tz_localize(None) if timestamp.tzinfo is not None else timestamp
        live = self._live_minute_history(timestamp, completed_history)
        seed = self.seed.loc[self.seed.index <= wall]
        minute = pd.concat((seed, live)).groupby(level=0).last().sort_index()
        features = self._features(wall, minute)
        forecast = np.maximum(
            np.asarray(self.model.predict(
                np.ascontiguousarray(features.to_numpy(dtype=np.float32))
            ), dtype=float), 0.0,
        )
        if forecast.shape != (HORIZON_STEPS,) or not np.isfinite(forecast).all():
            raise ValueError("Multiresolution RF forecast is incomplete")
        result = pd.Series(forecast, index=pd.date_range(timestamp, periods=HORIZON_STEPS, freq=INTERVAL),
                           name="outside_hub_with_resistance_forecast_kw")
        tail_source_day = None
        if steps == EXTENDED_STEPS:
            minute_start = minute.copy()
            minute_start.index = minute_start.index - MINUTE
            tail_source_day, source_day = prior_complete_local_day(
                minute_start.to_frame("outside_hub_with_resistance_kw"), timestamp,
            )
            tail_targets = pd.date_range(
                timestamp + RF_STEPS * INTERVAL,
                periods=EXTENDED_STEPS - RF_STEPS,
                freq=INTERVAL,
            )
            tail = pd.Series(
                np.maximum(clock_slot_forecast(
                    source_day, tail_targets,
                    column="outside_hub_with_resistance_kw",
                ), 0.0),
                index=tail_targets,
                name=result.name,
            )
            result = pd.concat((result, tail))
        # These are the actual RF feature values, after the feature builder's
        # availability handling. The prior 60 minute weather mean is complete
        # at issue time and is persisted across all 32 future target intervals.
        weather_row = features.iloc[0]
        result.attrs.update({
            "persisted_ambient_temperature_c": float(weather_row["future_ambient_temperature_c"]),
            "persisted_ambient_temperature_available": bool(
                weather_row["future_ambient_temperature_available"]
            ),
            "persisted_ambient_relative_humidity": float(
                weather_row["future_ambient_relative_humidity"]
            ),
            "persisted_ambient_relative_humidity_available": bool(
                weather_row["future_ambient_relative_humidity_available"]
            ),
            "weather_inputs_ready_at": timestamp,
            "persisted_weather_window_start": timestamp - HOUR,
            "persisted_weather_window_end_exclusive": timestamp,
            "tail_policy": self.tail_policy,
            "tail_source_day": None if tail_source_day is None else tail_source_day.isoformat(),
            "tail_first_step": RF_STEPS if tail_source_day is not None else None,
        })
        self.history.append({
            "timestamp": timestamp,
            "method": METHOD,
            "latest_completed_meter_at": completed_history.index[
                completed_history.index <= timestamp
            ][-1],
            "input_history_minutes": len(minute.loc[minute.index > wall - 24 * HOUR]),
            "seed_minutes_before_live": int((self.seed.index < live.index[0]).sum()),
            "forecast_steps": steps,
            "forecast_first_kw": float(forecast[0]),
            "forecast_max_kw": float(forecast.max()),
            "tail_source_day": None if tail_source_day is None else tail_source_day.isoformat(),
        })
        return result
