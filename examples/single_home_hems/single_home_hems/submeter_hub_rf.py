"""Direct 20-hour RF Hub forecasts with completed circuit power histories.

The separate aggregate-only provider remains unchanged. This provider uses
the evaluated 100-feature RF with four weeks of training, leaf size 15 and
uniform sample weights. Its companion RF retains the existing three eligible
load targets and their inverter/Hub consistency projection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.metadata
import json
from pathlib import Path
from typing import Any, ClassVar

import joblib
import numpy as np
import pandas as pd

from ..forecasting import evaluate_simulated_hub_submeter_rf as study
from .eligible_hub_rf import (
    ELIGIBLE_TARGETS, HUB_TARGET_DESCRIPTION, INVERTER_KW, RF_PARAMS,
    _local_naive, _series,
)
from .ev_hub_forecast import EVHubIntervalForecast
from .home_hub_observation import HUB_POWER_KEY
from .runtime_threads import apply_rf_threads


MODEL_NAME = "submeter_history_rf_4wk_20h_eligible_no_er"
CIRCUIT_COLUMNS = tuple(study.CIRCUIT_DEFINITIONS)
HORIZON_STEPS = study.HORIZON_STEPS
INTERVAL = study.INTERVAL
MINUTE = pd.Timedelta(minutes=1)
HISTORY_STEPS = 7 * 96
DEFAULT_ARTIFACT_DIR = study.BASELINE.parent / "ev_submeter_rf_4wk_20h_no_er"
DEFAULT_REFERENCE = (
    study.BASELINE / "submeter_rf_20h_recency_20261105/candidate_predictions"
    / "rf_submeter_history_4w_r1_leaf15.csv.gz"
)


def _circuit_frame(value: pd.DataFrame, label: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame) or set(value) != set(CIRCUIT_COLUMNS):
        raise ValueError(f"{label} requires exactly the three canonical circuit columns")
    return pd.DataFrame({name: _series(value[name], f"{label}: {name}", nonnegative=True)
                         for name in CIRCUIT_COLUMNS})


def _project(hub: np.ndarray, eligible: np.ndarray, inverter_kw: float) -> np.ndarray:
    """Preserve the existing uncapped mean, capped mean, and minimum projection."""
    if (hub.ndim != 1 or eligible.shape != (len(hub), 3)
            or not np.isfinite(hub).all() or not np.isfinite(eligible).all()):
        raise ValueError("RF outputs must be finite complete Hub and three-summary predictions")
    hub = np.maximum(hub, 0.0)
    eligible = np.maximum(eligible, 0.0)
    raw = np.minimum(eligible[:, 0], hub)
    capped = np.minimum.reduce((eligible[:, 1], raw, np.full(len(hub), inverter_kw)))
    minimum = np.minimum.reduce((eligible[:, 2], capped, np.full(len(hub), inverter_kw)))
    return np.column_stack((raw, capped, minimum))


def train_four_week_submeter_rf(
    meter_file: str | Path, private_labels_file: str | Path,
    observed_weather_file: str | Path, raw_circuits_file: str | Path,
    output_dir: str | Path, *, reference_predictions_file: str | Path = DEFAULT_REFERENCE,
) -> dict[str, Any]:
    """Fit the approved unweighted Hub RF and its existing eligibility companion."""
    paths = {name: Path(value).expanduser().resolve() for name, value in {
        "meter": meter_file, "private_labels": private_labels_file,
        "weather": observed_weather_file, "raw_circuits": raw_circuits_file,
        "reference_predictions": reference_predictions_file,
    }.items()}
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Model output directory already contains files: {output}")
    data = study.original.read_baseline(paths["meter"], weather_15min_path=paths["weather"])
    circuits, circuit_audit = study.read_submeter_power(paths["raw_circuits"], data)
    private = pd.read_csv(paths["private_labels"], parse_dates=["timestamp"]).set_index("timestamp")
    private.index = _local_naive(pd.DatetimeIndex(private.index))
    if not private.index.equals(data.index) or not set(ELIGIBLE_TARGETS).issubset(private):
        raise ValueError("Eligible labels do not match the corrected six-week Hub study")
    labels = private.loc[:, list(ELIGIBLE_TARGETS)].apply(pd.to_numeric, errors="raise")
    values = labels.to_numpy(float)
    hub = data["hub_non_ev_power_kw"].to_numpy(float)
    if (not np.isfinite(values).all() or (values < -1e-8).any()
            or (values[:, 2] > values[:, 1] + 1e-8).any()
            or (values[:, 1] > values[:, 0] + 1e-8).any()
            or (values[:, 1:] > INVERTER_KW + 1e-8).any()
            or not np.allclose(values[:, 0], hub, rtol=0, atol=1e-7)):
        raise ValueError("Eligible labels violate the existing resistance-excluded V2H bounds")
    if "hub_mean_kw" in private and not np.allclose(private.hub_mean_kw, hub, rtol=0, atol=1e-7):
        raise ValueError("Private Hub labels differ from the corrected aggregate meter")
    rows = study.make_supervised(data, circuits)
    cutoff = data.index[0] + 5 * study.WEEK
    fit_start, test_end = cutoff - 4 * study.WEEK, cutoff + study.WEEK
    fit, test = rows.mask_fit(fit_start, cutoff), rows.mask_evaluation(cutoff, test_end)
    if (rows.origin[fit].nunique() != 653 or fit.sum() != 653 * 80
            or rows.target[fit].max() + INTERVAL > cutoff
            or rows.origin[test].nunique() != 593):
        raise ValueError("Approved four-week fit or purged 20-hour evaluation grid changed")
    columns = rows.feature_sets[study.METHODS[1]]
    x_fit, x_test = rows.x.loc[fit, list(columns)], rows.x.loc[test, list(columns)]
    y_fit, y_test = labels.reindex(rows.target[fit]).to_numpy(float), labels.reindex(rows.target[test]).to_numpy(float)
    hub_model = study.original._model("rf_history", RF_PARAMS)
    eligible_model = study.original._model("rf_history", RF_PARAMS)
    # Omitting sample_weight retains the original uniform bootstrap RNG path.
    hub_model.fit(x_fit, rows.y[fit])
    eligible_model.fit(x_fit, y_fit)
    hub_test = np.maximum(np.asarray(hub_model.predict(x_test), dtype=float), 0.0)
    eligible_test = np.maximum(np.asarray(eligible_model.predict(x_test), dtype=float), 0.0)
    projected = _project(hub_test, eligible_test, INVERTER_KW)
    reference = pd.read_csv(paths["reference_predictions"], parse_dates=["origin", "target"])
    forecast = study.prediction_rows(rows, test, hub_test, study.METHODS[1], 4)
    keys = ["origin", "target", "lead"]
    expected = reference.sort_values(keys).reset_index(drop=True)
    actual = forecast.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual[keys], expected[keys])
    np.testing.assert_allclose(actual.actual_kw, expected.actual_kw, rtol=0, atol=1e-12)
    parity_error = float(np.abs(actual.forecast_kw - expected.forecast_kw).max())
    if parity_error > 1e-12:
        raise ValueError("New Hub RF does not reproduce the approved unweighted leaf-15 study fit")
    for position, name in enumerate(ELIGIBLE_TARGETS):
        forecast[f"predicted_{name}"] = eligible_test[:, position]
        forecast[f"deployed_{name}"] = projected[:, position]
        forecast[f"actual_{name}"] = y_test[:, position]
    metadata = {
        "model_name": MODEL_NAME, "hub_target": HUB_TARGET_DESCRIPTION,
        "description": "Unweighted four-week direct 20-hour Hub RF with aggregate plus three circuit power histories; paired existing eligibility summaries",
        "training_weeks": 4, "recency_multiplier": 1, "horizon_steps": HORIZON_STEPS,
        "configuration_selection": "User-selected unweighted leaf-size-15 circuit-history RF; retains the current four-week training window. The prior comparison's CV selected leaf size 5, not this deployed choice.",
        "supported_issue_horizon_hours": [8, 20], "direct_rf_horizon_hours": 20,
        "fit_start_inclusive": str(fit_start), "fit_end_exclusive": str(cutoff),
        "held_out_start": str(cutoff), "held_out_end_exclusive": str(test_end),
        "fit_pairs": int(fit.sum()), "fit_origin_count": int(rows.origin[fit].nunique()),
        "fit_origins_hourly": True, "last_fit_target_end": str(rows.target[fit].max() + INTERVAL),
        "held_out_origins": int(rows.origin[test].nunique()), "held_out_pairs": int(test.sum()),
        "rf_settings": {"n_estimators": 80, "max_features": 0.8, "max_samples": 0.8,
                        "bootstrap": True, "random_state": 42, "n_jobs": 4, **RF_PARAMS},
        "sample_weight": None, "input_feature_columns": list(columns), "input_feature_count": len(columns),
        "input_visibility": "completed Hub-minus-EV aggregate, completed water-heater/outdoor-unit/indoor-fan powers, and completed observed ambient temperature",
        "circuit_definitions": study.CIRCUIT_DEFINITIONS, "circuit_audit": circuit_audit,
        "eligible_target": "HVAC compressor/fan + HPWH electric + Hub-other; no resistance, Outside-Hub, or EV",
        "eligible_target_columns": list(ELIGIBLE_TARGETS), "inverter_kw": INVERTER_KW,
        "eligibility_projection": "raw=min(raw_hat,hub_hat); capped=min(capped_hat,raw,inverter); minimum=min(minimum_hat,capped,inverter), after nonnegative clipping",
        "source_files": {name: str(path) for name, path in paths.items()},
        "source_sha256": {name: study.sha256(path) for name, path in paths.items()},
        "implementation_sha256": {str(Path(path).resolve()): study.sha256(Path(path))
                                  for path in (__file__, study.__file__, study.original.__file__)},
        "reference_hub_forecast_max_abs_difference_kw": parity_error,
        "held_out_hub_mae_kw": float(np.abs(hub_test - rows.y[test]).mean()),
        "held_out_eligible_mae_kw": {name: float(np.abs(eligible_test[:, i] - y_test[:, i]).mean())
                                    for i, name in enumerate(ELIGIBLE_TARGETS)},
        "held_out_deployed_eligible_mae_kw": {name: float(np.abs(projected[:, i] - y_test[:, i]).mean())
                                             for i, name in enumerate(ELIGIBLE_TARGETS)},
        "held_out_score_note": "Descriptive reproduction on the existing seven-day benchmark, already examined in prior comparisons; no new untouched test claim or test-based tuning",
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "joblib")},
    }
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump({"hub_model": hub_model, "eligible_model": eligible_model,
                 "feature_columns": columns, "metadata": metadata}, output / "models.joblib")
    study.save_predictions(forecast, output / "descriptive_test_predictions.csv.gz")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


@dataclass
class SubmeterHubRFProvider:
    """Issue direct Hub and eligible-load forecasts from completed meter history."""

    requires_circuit_history: ClassVar[bool] = True
    hub_model: Any = field(repr=False)
    eligible_model: Any = field(repr=False)
    feature_columns: tuple[str, ...]
    seed_history: pd.Series = field(repr=False)
    seed_circuit_history: pd.DataFrame = field(repr=False)
    ambient_history: pd.Series = field(repr=False)
    ev_name: str
    metadata: dict[str, Any] = field(repr=False)
    history: list[dict[str, Any]] = field(default_factory=list, repr=False)

    @classmethod
    def from_artifact(
        cls, artifact: str | Path, *, seed_history: pd.Series,
        seed_circuit_history: pd.DataFrame, ambient_history: pd.Series, ev_name: str,
    ) -> "SubmeterHubRFProvider":
        saved = joblib.load(Path(artifact))
        required = {"hub_model", "eligible_model", "feature_columns", "metadata"}
        if not isinstance(saved, dict) or set(saved) != required:
            raise ValueError("Submeter Hub RF artifact has an unexpected schema")
        metadata = saved["metadata"]
        if (metadata.get("model_name") != MODEL_NAME
                or metadata.get("hub_target") != HUB_TARGET_DESCRIPTION
                or metadata.get("horizon_steps") != HORIZON_STEPS
                or metadata.get("training_weeks") != 4
                or metadata.get("recency_multiplier") != 1
                or metadata.get("rf_settings", {}).get("min_samples_leaf") != 15):
            raise ValueError("Submeter RF artifact differs from the approved target/features/model")
        columns = tuple(saved["feature_columns"])
        if len(columns) != 100 or list(columns) != metadata.get("input_feature_columns"):
            raise ValueError("Submeter Hub RF requires its evaluated 100-feature schema")
        seed = _series(seed_history, "causal pretest Hub seed", nonnegative=True)
        circuits = _circuit_frame(seed_circuit_history, "causal pretest circuit seed")
        if not seed.index.equals(circuits.index):
            raise ValueError("Hub and circuit seed intervals must align")
        ambient = _series(ambient_history, "observed ambient history", nonnegative=False)
        if not isinstance(ev_name, str) or not ev_name:
            raise ValueError("EV name is required for the separate submeter")
        apply_rf_threads(saved["hub_model"])
        apply_rf_threads(saved["eligible_model"])
        return cls(saved["hub_model"], saved["eligible_model"], columns,
                   seed, circuits, ambient, ev_name, dict(metadata))

    def _live_completed_bins(self, timestamp: pd.Timestamp,
                             completed_history: pd.DataFrame) -> pd.DataFrame:
        columns = ["hub_non_ev_power_kw", *CIRCUIT_COLUMNS]
        empty = pd.DataFrame(columns=columns, index=pd.DatetimeIndex([]), dtype=float)
        if completed_history.empty:
            return empty
        if not isinstance(completed_history.index, pd.DatetimeIndex):
            raise TypeError("Completed circuit observations need timestamped minute ends")
        index = _local_naive(completed_history.index)
        if (index.hasnans or index.has_duplicates or not index.is_monotonic_increasing
                or not index.equals(index.floor(MINUTE))):
            raise ValueError("Completed minute ends must be unique, increasing, and minute-aligned")
        wall = timestamp.tz_localize(None) if timestamp.tzinfo is not None else timestamp
        available = completed_history.loc[(index <= wall) & (index > wall - study.WEEK)].copy()
        if available.empty:
            return empty
        ev_key = f"{self.ev_name} Electric Power (kW)"
        missing = {HUB_POWER_KEY, ev_key, *CIRCUIT_COLUMNS} - set(available)
        if missing:
            raise ValueError(f"Completed submeter history lacks {sorted(missing)}")
        minute = available.loc[:, list(CIRCUIT_COLUMNS)].astype(float)
        minute["hub_non_ev_power_kw"] = available[HUB_POWER_KEY].astype(float) - available[ev_key].astype(float)
        # OCHRE controller history labels the end of each completed minute.
        minute.index = _local_naive(pd.DatetimeIndex(available.index)) - MINUTE
        if not np.isfinite(minute.to_numpy()).all() or minute.lt(-1e-7).any().any():
            raise ValueError("Completed non-EV Hub and circuit observations must be finite and nonnegative")
        if (minute.loc[:, list(CIRCUIT_COLUMNS)].sum(axis=1) > minute.hub_non_ev_power_kw + 1e-7).any():
            raise ValueError("Completed circuit powers exceed the resistance-excluded non-EV Hub")
        grouped = minute.resample("15min", closed="left", label="left")
        counts = grouped.count()
        if not counts.eq(15).all().all():
            raise ValueError("Live Hub/circuit history contains an incomplete completed 15-minute bin")
        return grouped.mean().loc[:, columns]

    def _issue_features(self, timestamp: pd.Timestamp, load: pd.Series,
                        circuits: pd.DataFrame, steps: int = HORIZON_STEPS) -> pd.DataFrame:
        wall = timestamp.tz_localize(None) if timestamp.tzinfo is not None else timestamp
        expected = pd.date_range(wall - study.WEEK, wall, freq=INTERVAL, inclusive="left")
        history = pd.concat([load.rename("hub_non_ev_power_kw"), circuits], axis=1).reindex(expected)
        if (history.isna().any().any() or not np.isfinite(history.to_numpy(float)).all()
                or len(history) != HISTORY_STEPS):
            raise ValueError("Submeter RF lacks complete causal seven-day Hub and circuit histories")
        previous = wall - INTERVAL
        if previous not in self.ambient_history.index:
            raise ValueError("Submeter RF lacks completed observed ambient temperature")
        # Appending only an unused placeholder lets the identical training
        # shift/rolling implementation compute the issue's strictly-past row.
        extended = history.reindex(history.index.append(pd.DatetimeIndex([wall])))
        parts = [study.power_history(extended.hub_non_ev_power_kw, "load")]
        parts.extend(study.power_history(extended[name], study.CIRCUIT_PREFIXES[name])
                     for name in CIRCUIT_COLUMNS)
        base = pd.concat(parts, axis=1).iloc[-1].to_dict()
        base["ambient_lag_1_c"] = float(self.ambient_history.loc[previous])
        base["ambient_lag_1_available"] = 1
        targets = pd.date_range(wall, periods=steps, freq=INTERVAL)
        clock = 2 * np.pi * (targets.hour * 60 + targets.minute) / 1440
        week = 2 * np.pi * targets.dayofweek / 7
        features = pd.DataFrame({name: np.repeat(value, steps) for name, value in base.items()})
        features["lead_step"] = np.arange(1, steps + 1)
        features["target_clock_sin"], features["target_clock_cos"] = np.sin(clock), np.cos(clock)
        features["target_week_sin"], features["target_week_cos"] = np.sin(week), np.cos(week)
        features["target_is_weekend"] = (targets.dayofweek >= 5).astype(int)
        if set(self.feature_columns) - set(features):
            raise ValueError("Submeter RF inference lacks training features")
        features = features.loc[:, list(self.feature_columns)]
        if not np.isfinite(features.to_numpy(float)).all():
            raise ValueError("Submeter RF issue features contain nonfinite values")
        return features

    def issue(self, timestamp: pd.Timestamp, completed_history: pd.DataFrame,
              horizon: pd.Timedelta, inverter_kw: float) -> EVHubIntervalForecast:
        timestamp, horizon = pd.Timestamp(timestamp), pd.Timedelta(horizon)
        if timestamp != timestamp.floor(INTERVAL) or horizon not in (32 * INTERVAL, 80 * INTERVAL):
            raise ValueError("Submeter RF requires aligned eight- or 20-hour issues")
        if not np.isclose(inverter_kw, self.metadata["inverter_kw"], rtol=0, atol=1e-9):
            raise ValueError("Submeter RF targets were trained for a different inverter")
        wall = timestamp.tz_localize(None) if timestamp.tzinfo is not None else timestamp
        seed = pd.concat([self.seed_history.rename("hub_non_ev_power_kw"), self.seed_circuit_history], axis=1)
        seed = seed.loc[seed.index < wall]
        live = self._live_completed_bins(timestamp, completed_history)
        merged = pd.concat([seed, live]).groupby(level=0).last().sort_index()
        steps = int(horizon / INTERVAL)
        features = self._issue_features(timestamp, merged.hub_non_ev_power_kw,
                                        merged.loc[:, list(CIRCUIT_COLUMNS)], steps)
        hub = np.maximum(np.asarray(self.hub_model.predict(features), dtype=float), 0.0)
        eligible = np.maximum(np.asarray(self.eligible_model.predict(features), dtype=float), 0.0)
        if hub.shape != (steps,):
            raise ValueError("Submeter RF returned an incomplete Hub forecast")
        projected = _project(hub, eligible, inverter_kw)
        frame = pd.DataFrame(projected, columns=ELIGIBLE_TARGETS,
                             index=pd.date_range(timestamp, periods=steps, freq=INTERVAL))
        frame["hub_mean_kw"] = hub
        issued = EVHubIntervalForecast(timestamp, timestamp, frame, MODEL_NAME)
        self.history.append({
            "issued_at": timestamp,
            "latest_live_complete_bin": live.index.max() if not live.empty else pd.NaT,
            "latest_seed_bin": seed.index.max() if not seed.empty else pd.NaT,
            "eligible_raw_clipped_to_hub_count": int(np.sum(eligible[:, 0] > hub + 1e-9)),
            "eligible_raw_clipped_to_hub_max_kw": float(np.maximum(eligible[:, 0] - hub, 0).max()),
            "forecast_steps": len(frame), "direct_rf_steps": len(frame), "tail_source_day": None,
            "circuit_history_required": True,
        })
        return issued
