"""Four-week plain-history RF forecasts for the EV-only Home Hub controller.

The Hub meter excludes HVAC electric resistance. One RF predicts the complete
non-EV Hub mean for site power. A second RF with the same features,
hyperparameters, and fit window predicts three summaries of V2H-eligible load:
heat-pump compressor/fan, water heating, and Hub-other. Their raw mean targets
are physically equal in this wiring, while the latter two eligible summaries
apply the minute-level inverter clip. Private circuit labels are used only
during the frozen pretest fit. At issuance both forests see completed aggregate
Hub and EV meters plus past observed ambient temperature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .runtime_threads import apply_rf_threads

from .ev_hub_forecast import EVHubIntervalForecast
from .home_hub_observation import HUB_POWER_KEY
from .prior_day_load_forecast import (
    EXTENDED_STEPS, INTERVAL, MINUTE, RF_STEPS, prior_complete_local_day,
)


HORIZON_STEPS = RF_STEPS
WEEK = pd.Timedelta(days=7)


ELIGIBLE_TARGETS = (
    "eligible_load_mean_kw", "eligible_mean_kw", "eligible_min_kw",
)
RF_PARAMS = {"min_samples_leaf": 15}
INVERTER_KW = 9.6
MODEL_NAME = "plain_history_rf_4wk_eligible_no_er"
HUB_TARGET_DESCRIPTION = "15-minute non-EV Home Hub mean, excluding HVAC resistance"
_SEED_PERIODS = 7 * 96


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _local_naive(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return index.tz_localize(None) if index.tz is not None else index


def _series(value: pd.Series, label: str, *, nonnegative: bool) -> pd.Series:
    if not isinstance(value, pd.Series) or not isinstance(value.index, pd.DatetimeIndex):
        raise TypeError(f"{label} must be a timestamp-indexed Series")
    result = value.astype(float).copy()
    result.index = _local_naive(result.index)
    if (result.index.hasnans or result.index.has_duplicates
            or not result.index.is_monotonic_increasing
            or not result.index.equals(result.index.floor(INTERVAL))):
        raise ValueError(f"{label} must have unique increasing 15-minute local interval starts")
    if not np.isfinite(result.to_numpy()).all() or (nonnegative and (result < -1e-8).any()):
        raise ValueError(f"{label} has missing, nonfinite, or negative values")
    return result


def train_four_week_eligible_rf(
    meter_file: str | Path,
    private_labels_file: str | Path,
    observed_weather_file: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Reproduce the original week-six RF fit with the corrected backing target.

    The four-week final fit uses study weeks 2--5.  Week 5 selected RF leaf
    size 15 in the original validation study, and week 6 remains excluded from
    fitting.  Both models have the original hourly origin stride and purged
    complete eight-hour targets.
    """
    from ..forecasting.evaluate_simulated_home_hub import (
        _model, make_supervised, read_baseline,
    )

    meter_path = Path(meter_file).expanduser().resolve()
    label_path = Path(private_labels_file).expanduser().resolve()
    weather_path = Path(observed_weather_file).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    data = read_baseline(meter_path, weather_15min_path=weather_path)
    private = pd.read_csv(label_path, parse_dates=["timestamp"]).set_index("timestamp")
    private.index = _local_naive(pd.DatetimeIndex(private.index))
    if not private.index.equals(data.index) or not set(ELIGIBLE_TARGETS).issubset(private):
        raise ValueError("Private eligible labels do not match the six-week Hub study")
    labels = private.loc[:, list(ELIGIBLE_TARGETS)].apply(pd.to_numeric, errors="raise")
    values = labels.to_numpy(float)
    hub = data["hub_non_ev_power_kw"].to_numpy(float)
    if (not np.isfinite(values).all() or (values < -1e-8).any()
            or (values[:, 2] > values[:, 1] + 1e-8).any()
            or (values[:, 1] > values[:, 0] + 1e-8).any()
            or (values[:, 0] > hub + 1e-7).any()
            or (values[:, 1:] > INVERTER_KW + 1e-8).any()):
        raise ValueError("Private eligible labels violate the agreed V2H circuit bounds")
    if not np.allclose(values[:, 0], hub, rtol=0, atol=1e-7):
        raise ValueError("Resistance-excluded Hub must equal uncapped eligible load")
    if "hub_mean_kw" in private and not np.allclose(
        private["hub_mean_kw"].to_numpy(float), hub, rtol=0, atol=1e-7,
    ):
        raise ValueError("Private Hub labels differ from the aggregate meter")

    rows = make_supervised(data)
    study_start = pd.Timestamp(data.index[0])
    test_start = study_start + 5 * WEEK
    test_end = study_start + 6 * WEEK
    fit_start = test_start - 4 * WEEK
    fit = rows.mask_fit(fit_start, test_start)
    test = rows.mask_evaluation(test_start, test_end)
    columns = rows.feature_sets["history"]
    if (not fit.any() or not test.any() or rows.origin[fit].min() < fit_start
            or rows.target[fit].max() >= test_start):
        raise ValueError("The four-week RF fit leaks into the held-out week")
    x_fit = rows.x.loc[fit, list(columns)]
    x_test = rows.x.loc[test, list(columns)]
    y_fit = labels.reindex(rows.target[fit]).to_numpy(float)
    y_test = labels.reindex(rows.target[test]).to_numpy(float)
    if not np.isfinite(y_fit).all() or not np.isfinite(y_test).all():
        raise ValueError("Eligible fit or held-out labels are incomplete")

    hub_model = _model("rf_history", RF_PARAMS)
    eligible_model = _model("rf_history", RF_PARAMS)
    hub_model.fit(x_fit, rows.y[fit])
    eligible_model.fit(x_fit, y_fit)
    hub_test = np.maximum(np.asarray(hub_model.predict(x_test), dtype=float), 0.0)
    eligible_test = np.maximum(np.asarray(eligible_model.predict(x_test), dtype=float), 0.0)
    if eligible_test.shape != (len(hub_test), len(ELIGIBLE_TARGETS)):
        raise ValueError("Eligible RF output is not the three expected summaries")
    consistency = eligible_test[:, 0] > hub_test + 1e-9
    deployed_raw = np.minimum(eligible_test[:, 0], hub_test)
    deployed_capped = np.minimum.reduce((eligible_test[:, 1], deployed_raw,
                                         np.full(len(hub_test), INVERTER_KW)))
    deployed_min = np.minimum.reduce((eligible_test[:, 2], deployed_capped,
                                      np.full(len(hub_test), INVERTER_KW)))
    deployed_eligible = np.column_stack((deployed_raw, deployed_capped, deployed_min))
    metadata: dict[str, Any] = {
        "model_name": MODEL_NAME,
        "description": "Non-EV Hub RF plus same-architecture multi-output RF for V2H-eligible load with resistance outside the Hub",
        "fit_start_inclusive": fit_start.isoformat(),
        "fit_end_exclusive": test_start.isoformat(),
        "held_out_start": test_start.isoformat(),
        "held_out_end_exclusive": test_end.isoformat(),
        "fit_origins_hourly": True,
        "fit_pairs": int(fit.sum()),
        "fit_origin_count": int(rows.origin[fit].nunique()),
        "rf_settings": {"n_estimators": 80, "max_features": 0.8,
                        "max_samples": 0.8, "bootstrap": True,
                        "random_state": 42, **RF_PARAMS},
        "input_feature_columns": list(columns),
        "input_visibility": "completed aggregate non-EV Hub and past observed ambient only",
        "hub_target": HUB_TARGET_DESCRIPTION,
        "eligible_target": "HVAC compressor/fan + HPWH electric + Hub-other; no resistance, Outside-Hub, or EV",
        "eligible_target_columns": list(ELIGIBLE_TARGETS),
        "inverter_kw": INVERTER_KW,
        "meter_file": str(meter_path),
        "meter_sha256": _sha256(meter_path),
        "private_labels_file": str(label_path),
        "private_labels_sha256": _sha256(label_path),
        "observed_weather_file": str(weather_path),
        "observed_weather_sha256": _sha256(weather_path),
        "held_out_hub_mae_kw": float(np.mean(np.abs(hub_test - rows.y[test]))),
        "held_out_eligible_mae_kw": {
            name: float(np.mean(np.abs(eligible_test[:, i] - y_test[:, i])))
            for i, name in enumerate(ELIGIBLE_TARGETS)
        },
        "held_out_deployed_eligible_mae_kw": {
            name: float(np.mean(np.abs(deployed_eligible[:, i] - y_test[:, i])))
            for i, name in enumerate(ELIGIBLE_TARGETS)
        },
        "held_out_eligible_exceeds_hub_forecast_rows": int(consistency.sum()),
        "held_out_eligible_exceeds_hub_forecast_max_kw": float(
            np.maximum(eligible_test[:, 0] - hub_test, 0.0).max()
        ),
        "held_out_score_note": "Descriptive only; week 6 was already used to compare candidate methods",
    }
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Model output directory already contains files: {output}")
    output.mkdir(parents=True)
    joblib.dump({"hub_model": hub_model, "eligible_model": eligible_model,
                 "feature_columns": columns, "metadata": metadata}, output / "models.joblib")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


@dataclass
class EligibleHubRFProvider:
    """Issue both Hub/site and eligible/backing forecasts without future data."""

    hub_model: Any = field(repr=False)
    eligible_model: Any = field(repr=False)
    feature_columns: tuple[str, ...]
    seed_history: pd.Series = field(repr=False)
    ambient_history: pd.Series = field(repr=False)
    ev_name: str
    metadata: dict[str, Any] = field(repr=False)
    tail_policy: str = "none"
    history: list[dict[str, Any]] = field(default_factory=list, repr=False)

    @classmethod
    def from_artifact(
        cls, artifact: str | Path, *, seed_history: pd.Series,
        ambient_history: pd.Series, ev_name: str,
        tail_policy: str = "none",
    ) -> "EligibleHubRFProvider":
        saved = joblib.load(Path(artifact))
        required = {"hub_model", "eligible_model", "feature_columns", "metadata"}
        if not isinstance(saved, dict) or set(saved) != required:
            raise ValueError("Eligible RF artifact has an unexpected schema")
        if saved["metadata"].get("hub_target") != HUB_TARGET_DESCRIPTION:
            raise ValueError("Eligible RF artifact uses a different Home Hub wiring")
        apply_rf_threads(saved["hub_model"])
        apply_rf_threads(saved["eligible_model"])
        seed = _series(seed_history, "causal pretest Hub seed", nonnegative=True)
        ambient = _series(ambient_history, "observed ambient history", nonnegative=False)
        if not isinstance(ev_name, str) or not ev_name:
            raise ValueError("EV name is required for the separate submeter")
        if tail_policy not in {"none", "prior_complete_local_day"}:
            raise ValueError("Unknown Home Hub forecast tail policy")
        metadata = dict(saved["metadata"])
        if tail_policy == "prior_complete_local_day":
            metadata.update(tail_policy=tail_policy, tail_first_step=RF_STEPS,
                            tail_horizon_steps=EXTENDED_STEPS)
        return cls(saved["hub_model"], saved["eligible_model"],
                   tuple(saved["feature_columns"]), seed, ambient,
                   ev_name, metadata, tail_policy)

    def _prior_day_tail(
        self, timestamp: pd.Timestamp, completed_history: pd.DataFrame,
        inverter_kw: float,
    ) -> tuple[pd.Timestamp, pd.DataFrame]:
        """Use only completed aggregate meter minutes from an earlier local day."""
        ev_key = f"{self.ev_name} Electric Power (kW)"
        if not {HUB_POWER_KEY, ev_key}.issubset(completed_history):
            raise ValueError("Prior-day Hub tail needs completed Hub and EV meters")
        available = completed_history.loc[
            completed_history.index <= timestamp, [HUB_POWER_KEY, ev_key]
        ]
        minute = available[HUB_POWER_KEY].astype(float) - available[ev_key].astype(float)
        minute.index = pd.DatetimeIndex(minute.index) - MINUTE
        source_day, rows = prior_complete_local_day(
            minute.to_frame("hub_non_ev_kw"), timestamp,
        )
        power = np.maximum(rows["hub_non_ev_kw"].to_numpy(dtype=float), 0.0)
        raw = power.reshape(96, 15).mean(axis=1)
        clipped = np.minimum(power, inverter_kw).reshape(96, 15)
        capped = clipped.mean(axis=1)
        minimum = clipped.min(axis=1)
        targets = pd.date_range(timestamp + RF_STEPS * INTERVAL,
                                periods=EXTENDED_STEPS - RF_STEPS, freq=INTERVAL)
        slots = np.asarray(targets.hour * 4 + targets.minute // 15, dtype=int)
        tail = pd.DataFrame({
            "hub_mean_kw": raw[slots],
            "eligible_load_mean_kw": raw[slots],
            "eligible_mean_kw": capped[slots],
            "eligible_min_kw": minimum[slots],
        }, index=targets)
        return source_day, tail

    def _live_completed_bins(
        self, timestamp: pd.Timestamp, completed_history: pd.DataFrame,
    ) -> pd.Series:
        if completed_history.empty:
            return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
        if not isinstance(completed_history.index, pd.DatetimeIndex):
            raise TypeError("Completed Hub observations need timestamps")
        if completed_history.index.has_duplicates or not completed_history.index.is_monotonic_increasing:
            raise ValueError("Completed Hub timestamps must be unique and increasing")
        ev_key = f"{self.ev_name} Electric Power (kW)"
        if not {HUB_POWER_KEY, ev_key}.issubset(completed_history):
            raise ValueError("Completed history lacks aggregate Hub or EV meter")
        available = completed_history.loc[completed_history.index <= timestamp, [HUB_POWER_KEY, ev_key]]
        if available.empty:
            return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
        minute = (available[HUB_POWER_KEY].astype(float) - available[ev_key].astype(float))
        minute.index = _local_naive(pd.DatetimeIndex(minute.index))
        values = minute.to_numpy(dtype=float)
        if (np.isinf(values).any() or np.any(values[np.isfinite(values)] < -1e-7)
                or (self.tail_policy == "none" and not np.isfinite(values).all())):
            raise ValueError("Completed non-EV Hub meter contains invalid power")
        grouped = minute.resample("15min", closed="right", label="right")
        mean, count = grouped.mean(), grouped.count()
        complete = mean.loc[count.eq(15)].copy()
        complete.index -= INTERVAL
        complete.name = "hub_non_ev_power_kw"
        return complete

    def _issue_features(self, timestamp: pd.Timestamp, load: pd.Series) -> pd.DataFrame:
        """Reproduce make_supervised's plain-history features for one origin."""
        time = timestamp.tz_localize(None) if timestamp.tzinfo is not None else timestamp
        previous = time - INTERVAL
        expected = pd.date_range(time - _SEED_PERIODS * INTERVAL, time, freq=INTERVAL,
                                 inclusive="left")
        history = load.reindex(expected)
        if history.isna().any() or not np.isfinite(history.to_numpy(float)).all():
            raise ValueError("RF forecast lacks a complete causal seven-day Hub history")
        ambient_time = previous
        if ambient_time not in self.ambient_history.index:
            raise ValueError("RF forecast lacks completed observed ambient temperature")
        ambient = float(self.ambient_history.loc[ambient_time])
        base: dict[str, float | int] = {}
        for lag in (1, 2, 4, 8, 96, 672):
            base[f"load_lag_{lag}_kw"] = float(history.iloc[-lag])
            base[f"load_lag_{lag}_available"] = 1
        for steps in (4, 8, 96):
            values = history.iloc[-steps:]
            base[f"load_mean_prev_{steps}_kw"] = float(values.mean())
            base[f"load_max_prev_{steps}_kw"] = float(values.max())
            base[f"load_count_prev_{steps}"] = steps
        base["load_slope_1h_kw"] = float(history.iloc[-1] - history.iloc[-5])
        base["load_slope_1h_available"] = 1
        base["ambient_lag_1_c"] = ambient
        base["ambient_lag_1_available"] = 1
        targets = pd.date_range(time, periods=HORIZON_STEPS, freq=INTERVAL)
        clock = 2 * np.pi * (targets.hour * 60 + targets.minute) / 1440
        week = 2 * np.pi * targets.dayofweek / 7
        features = pd.DataFrame({key: np.repeat(value, HORIZON_STEPS)
                                 for key, value in base.items()})
        features["lead_step"] = np.arange(1, HORIZON_STEPS + 1)
        features["target_clock_sin"] = np.sin(clock)
        features["target_clock_cos"] = np.cos(clock)
        features["target_week_sin"] = np.sin(week)
        features["target_week_cos"] = np.cos(week)
        features["target_is_weekend"] = (targets.dayofweek >= 5).astype(int)
        missing = set(self.feature_columns) - set(features)
        if missing:
            raise ValueError(f"RF inference lacks training features: {sorted(missing)}")
        return features.loc[:, list(self.feature_columns)]

    def issue(
        self, timestamp: pd.Timestamp, completed_history: pd.DataFrame,
        horizon: pd.Timedelta, inverter_kw: float,
    ) -> EVHubIntervalForecast:
        timestamp = pd.Timestamp(timestamp)
        horizon = pd.Timedelta(horizon)
        allowed = {HORIZON_STEPS * INTERVAL}
        if self.tail_policy == "prior_complete_local_day":
            allowed.add(EXTENDED_STEPS * INTERVAL)
        if timestamp != timestamp.floor(INTERVAL) or horizon not in allowed:
            raise ValueError("Eligible RF requires an aligned supported eight- or 20-hour issue")
        if not np.isclose(inverter_kw, self.metadata["inverter_kw"], rtol=0, atol=1e-9):
            raise ValueError("Eligible RF targets were trained for a different inverter")
        wall = timestamp.tz_localize(None) if timestamp.tzinfo is not None else timestamp
        seed = self.seed_history.loc[self.seed_history.index < wall]
        live = self._live_completed_bins(timestamp, completed_history)
        live = live.loc[live.index < wall]
        load = pd.concat([seed, live]).groupby(level=0).last().sort_index()
        x = self._issue_features(timestamp, load)
        hub = np.maximum(np.asarray(self.hub_model.predict(x), dtype=float), 0.0)
        eligible = np.maximum(np.asarray(self.eligible_model.predict(x), dtype=float), 0.0)
        if hub.shape != (HORIZON_STEPS,) or eligible.shape != (HORIZON_STEPS, 3):
            raise ValueError("RF issue returned incomplete horizon predictions")
        raw_unprojected = eligible[:, 0].copy()
        raw = np.minimum(eligible[:, 0], hub)
        capped = np.minimum.reduce((eligible[:, 1], raw,
                                    np.full(HORIZON_STEPS, inverter_kw)))
        minimum = np.minimum.reduce((eligible[:, 2], capped,
                                     np.full(HORIZON_STEPS, inverter_kw)))
        if not np.isfinite(np.column_stack((hub, raw, capped, minimum))).all():
            raise ValueError("RF issue returned nonfinite predictions")
        starts = pd.date_range(timestamp, periods=HORIZON_STEPS, freq=INTERVAL)
        frame = pd.DataFrame({
            "hub_mean_kw": hub,
            "eligible_load_mean_kw": raw,
            "eligible_mean_kw": capped,
            "eligible_min_kw": minimum,
        }, index=starts)
        tail_source_day = None
        if horizon == EXTENDED_STEPS * INTERVAL:
            tail_source_day, tail = self._prior_day_tail(
                timestamp, completed_history, inverter_kw,
            )
            frame = pd.concat((frame, tail))
        issued = EVHubIntervalForecast(
            issued_at=timestamp, inputs_ready_at=timestamp,
            intervals=frame,
            source=(MODEL_NAME if tail_source_day is None else
                    f"{MODEL_NAME}__prior_complete_local_day_after_8h"),
        )
        self.history.append({
            "issued_at": timestamp,
            "latest_live_complete_bin": live.index.max() if not live.empty else pd.NaT,
            "latest_seed_bin": seed.index.max() if not seed.empty else pd.NaT,
            "eligible_raw_clipped_to_hub_count": int(np.sum(raw_unprojected > hub + 1e-9)),
            "eligible_raw_clipped_to_hub_max_kw": float(np.maximum(raw_unprojected - hub, 0).max()),
            "forecast_steps": len(frame),
            "tail_source_day": None if tail_source_day is None else tail_source_day.isoformat(),
        })
        return issued
