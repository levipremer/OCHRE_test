"""Causal DC House rest-of-home forecast for the joint single-home HEMS.

The first eight 15-minute bins use the established conservative random forest
(``rf_q75_peak``). Later bins use the 90th percentile of each clock slot over
14 or 20 completed source calendar days. This is a hard switch after two hours;
the forecast is an exogenous power input, not an optimizer decision variable.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from ..forecasting.evaluate_rest_of_home import (
    INTERVAL,
    PEAK_REFERENCE_END,
    PEAK_REFERENCE_START,
    STEPS,
    TRAIN_DAYS,
    conservative_peak_forecast,
    features_from_observed,
    fit_forest,
    fit_nonpeak_residual_quantiles,
    fit_peak_continuation,
    observed_peak_run_age,
    read_15_minute_power,
)


SHORT_STEPS = 8
PERCENTILE = 0.90
POWER_NAME = "rest_of_home_forecast_kw"
INFERENCE_GAP_POLICIES = ("strict", "causal_short_gap")


def _wall_time(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("Rest-of-home time cannot be missing")
    return timestamp.tz_localize(None) if timestamp.tz is not None else timestamp


def _source_start(value: str | pd.Timestamp, simulation_start: pd.Timestamp) -> pd.Timestamp:
    source = pd.Timestamp(value)
    if pd.isna(source) or source.tz is not None:
        raise ValueError("Rest-of-home source_start must be a naive local timestamp")
    if isinstance(value, str) and len(value) == 10:
        source += simulation_start - simulation_start.normalize()
    if source.time() != simulation_start.time():
        raise ValueError("Rest-of-home source and simulation starts must have the same local clock time")
    return source


def _p90_clock_profile(
    approved: pd.Series, observed: pd.Series, source_origin: pd.Timestamp,
    history_days: int,
) -> pd.Series:
    """Match the validation p90 sweep, including its final-hour fill guard."""
    issue_day = source_origin.normalize()
    start = issue_day - pd.Timedelta(days=history_days)
    history = approved.loc[(approved.index >= start) & (approved.index < issue_day)].copy()
    if len(history) != history_days * 96:
        raise ValueError(f"Incomplete {history_days}-day p90 calendar history at {source_origin}")
    recent = history.index >= issue_day - pd.Timedelta(hours=1)
    history.loc[recent & observed.reindex(history.index).isna()] = np.nan
    clock_slot = history.index.hour * 4 + history.index.minute // 15
    profile = history.groupby(clock_slot).quantile(PERCENTILE, interpolation="linear")
    profile = profile.reindex(range(96))
    if profile.isna().any():
        raise ValueError(f"The {history_days}-day p90 history lacks a clock slot at {source_origin}")
    return profile


def _approved_bin_ready_times(
    path: Path, approved: pd.Series, observed: pd.Series,
) -> pd.Series:
    """Time each approved bin's last required raw observation became available.

    A linearly filled minute needs the next originally observed minute that
    closes its gap. An offline fill must not be used at an issue before that
    closing reading has arrived.
    """
    frame = pd.read_csv(path, usecols=["Time", "Value", "Quality"], parse_dates=["Time"])
    times = pd.DatetimeIndex(frame["Time"])
    quality = frame["Quality"].astype(str).to_numpy()
    present = pd.to_numeric(frame["Value"], errors="raise").notna().to_numpy()
    observed_mask = (quality == "observed") & present
    interpolated_mask = quality == "interpolated"
    raw_observed_time = pd.Series(pd.NaT, index=times, dtype="datetime64[ns]")
    raw_observed_time.iloc[np.flatnonzero(observed_mask)] = times[observed_mask]
    next_observed_time = raw_observed_time.bfill()
    minute_ready = pd.Series(times, index=times)
    minute_ready.iloc[np.flatnonzero(interpolated_mask)] = next_observed_time.iloc[
        np.flatnonzero(interpolated_mask)
    ].to_numpy()
    ready = minute_ready.resample("15min", closed="right", label="right").max()
    ready.index = ready.index - INTERVAL
    ready = ready.reindex(approved.index)
    ready.loc[approved.isna()] = pd.NaT
    if ready.loc[approved.notna()].isna().any():
        raise ValueError("Approved rest-of-home fill lacks a closing observed reading")
    observed_ready = ready.loc[observed.notna()]
    if not (observed_ready.to_numpy() == (observed_ready.index + INTERVAL).to_numpy()).all():
        raise ValueError("Originally observed bins have inconsistent availability times")
    return ready


def _causal_short_gap_inference_data(
    path: Path, simulation_start: pd.Timestamp, source_start: pd.Timestamp,
    source_window: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare inference-only fills from the already-approved plant policy.

    These copies never enter the training/calibration or p90 history. Minute
    availability is retained so a retrospective linear fill cannot leak its
    closing observation into an earlier forecast.
    """
    from examples.five_home_block.five_home_block.config import RestOfHomeConfig
    from examples.five_home_block.five_home_block.rest_of_home import RestOfHomeReplay

    if source_window is None:
        replay = RestOfHomeReplay.from_config(
            RestOfHomeConfig(
                source="dc_house_test", data_file=path,
                source_start=source_start.isoformat(),
                missing_data_policy="linear_interpolate",
            ),
            simulation_start=simulation_start,
        )
        fills = replay.interpolated_minutes.set_index("source_interval_start")
    else:
        # The original replay class intentionally restricts itself to the
        # September 15–28 test split. Apply its same bounded 1–4-minute rule
        # on an explicitly supplied source window for the six-week study.
        start, end = map(pd.Timestamp, source_window)
        if (start.tz is not None or end.tz is not None or start >= end
                or start != start.floor("min") or end != end.floor("min")
                or not start <= source_start < end):
            raise ValueError("Causal inference source window must cover source_start")
        window = pd.read_csv(path, usecols=["Time", "Value", "Quality"],
                             parse_dates=["Time"]).set_index("Time")
        window.index -= pd.Timedelta(minutes=1)
        window = window.reindex(pd.date_range(start, end, freq="min", inclusive="left"))
        values = pd.to_numeric(window["Value"], errors="raise").astype(float)
        approved = window["Quality"].isin(("observed", "interpolated"))
        values = values.where(approved)
        missing = values.isna()
        run_lengths = missing.groupby((~missing).cumsum()).transform("sum")
        filled = (values / 1000.0).interpolate(method="time", limit_area="inside")
        use_fill = missing & run_lengths.le(4) & filled.notna()
        fills = pd.DataFrame({"rest_of_home_power_kw": filled.loc[use_fill]})
        fills.index.name = "source_interval_start"
    raw = pd.read_csv(path, usecols=["Time", "Value", "Quality"], parse_dates=["Time"])
    raw = raw.set_index("Time")
    observed = raw["Quality"].eq("observed") & raw["Value"].notna()
    approved = raw["Quality"].isin(("observed", "interpolated"))
    next_observed = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")
    next_observed.loc[observed] = raw.index[observed]
    next_observed = next_observed.bfill()
    minutes = pd.DataFrame({
        "power_w": raw["Value"].where(approved),
        "observed_w": raw["Value"].where(observed),
        "ready_at": pd.Series(raw.index, index=raw.index),
        "plant_filled": False,
    })
    interpolated = raw["Quality"].eq("interpolated")
    minutes.loc[interpolated, "ready_at"] = next_observed.loc[interpolated]
    fill_ends = fills.index + pd.Timedelta(minutes=1)
    minutes.loc[fill_ends, "power_w"] = fills["rest_of_home_power_kw"].to_numpy() * 1000.0
    minutes.loc[fill_ends, "ready_at"] = next_observed.loc[fill_ends]
    minutes.loc[fill_ends, "plant_filled"] = True
    minutes.loc[minutes["power_w"].isna(), "ready_at"] = pd.NaT

    grouped = minutes.resample("15min", closed="right", label="right")
    complete = grouped["power_w"].count().eq(15) & grouped["ready_at"].count().eq(15)
    bins = pd.DataFrame({
        "power_w": grouped["power_w"].mean().where(complete),
        "ready_at": grouped["ready_at"].max().where(complete),
        "plant_filled_minutes": grouped["plant_filled"].sum(),
    })
    bins.index -= INTERVAL
    return bins, minutes


@dataclass
class RestOfHomeHybridForecast:
    """Fixed-fit RF and causal rolling p90 profile on the DC House source clock."""

    source_path: Path
    simulation_start: pd.Timestamp
    source_start: pd.Timestamp
    history_days: int
    approved: pd.Series = field(repr=False)
    observed: pd.Series = field(repr=False)
    approved_ready_at: pd.Series = field(repr=False)
    forest: RandomForestRegressor = field(repr=False)
    q75_residual_w: np.ndarray = field(repr=False)
    continuation: dict[str, np.ndarray] = field(repr=False)
    peak_threshold_w: float
    metadata: dict[str, Any] = field(repr=False)
    inference_gap_policy: str = "strict"
    _inference_bins: pd.DataFrame | None = field(default=None, repr=False)
    _inference_minutes: pd.DataFrame | None = field(default=None, repr=False)
    _daily_profiles: dict[pd.Timestamp, pd.Series] = field(
        default_factory=dict, init=False, repr=False
    )

    @property
    def method_label(self) -> str:
        return f"conservative_rf_first_2h__p90_{self.history_days}d_remaining"

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        simulation_start: str | pd.Timestamp,
        source_start: str | pd.Timestamp,
        history_days: int,
        inference_gap_policy: str = "strict",
        inference_window: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    ) -> "RestOfHomeHybridForecast":
        """Fit once using data strictly before the source replay begins.

        A date-only ``source_start`` inherits the simulation start's local
        clock time, matching the measured-load replay. The fixed peak threshold
        always comes from originally observed August 2--21, 2026 bins.
        """
        if type(history_days) is not int or history_days not in (14, 20):
            raise ValueError("history_days must be explicitly set to 14 or 20")
        if inference_gap_policy not in INFERENCE_GAP_POLICIES:
            raise ValueError(f"inference_gap_policy must be one of {INFERENCE_GAP_POLICIES}")
        if inference_window is not None and inference_gap_policy != "causal_short_gap":
            raise ValueError("inference_window requires causal_short_gap inference")
        source_path = Path(path).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Rest-of-home source CSV not found: {source_path}")
        sim_start = _wall_time(simulation_start)
        source_start_time = _source_start(source_start, sim_start)
        if sim_start != sim_start.floor("15min") or source_start_time != source_start_time.floor("15min"):
            raise ValueError("Rest-of-home simulation and source starts must align to 15-minute boundaries")
        if source_start_time < PEAK_REFERENCE_END:
            raise ValueError("Rest-of-home source replay must begin after the fixed peak-reference period")

        approved, observed, source_details = read_15_minute_power(source_path)
        approved_ready_at = _approved_bin_ready_times(source_path, approved, observed)
        inference_bins = inference_minutes = None
        if inference_gap_policy == "causal_short_gap":
            inference_bins, inference_minutes = _causal_short_gap_inference_data(
                source_path, sim_start, source_start_time, inference_window,
            )
        train_start = source_start_time - pd.Timedelta(days=TRAIN_DAYS)
        if observed.index[0] > min(PEAK_REFERENCE_START, train_start):
            raise ValueError("Rest-of-home source lacks the peak-reference or training history")
        if source_start_time not in observed.index or source_start_time - INTERVAL not in observed.index:
            raise ValueError("Rest-of-home source replay start is absent from the 15-minute grid")
        reference = observed.loc[
            (observed.index >= PEAK_REFERENCE_START)
            & (observed.index < PEAK_REFERENCE_END)
        ].dropna()
        if len(reference) < 100:
            raise ValueError("Insufficient originally observed pre-validation peak-reference bins")
        threshold_w = float(reference.quantile(0.95, interpolation="linear"))
        q75_offsets, q75_details = fit_nonpeak_residual_quantiles(
            observed, source_start_time, threshold_w
        )
        continuation, continuation_details = fit_peak_continuation(
            observed, source_start_time, threshold_w
        )
        forest, forest_details = fit_forest(observed, source_start_time)
        if q75_offsets.shape != (STEPS,) or any(
            curve.shape != (STEPS,) for curve in continuation.values()
        ):
            raise ValueError("Conservative RF calibration does not cover all 48 leads")
        metadata = {
            "method_label": f"conservative_rf_first_2h__p90_{history_days}d_remaining",
            "source_path": str(source_path),
            "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "simulation_start": sim_start.isoformat(),
            "source_start": source_start_time.isoformat(),
            "training_start_inclusive": train_start.isoformat(),
            "training_end_exclusive": source_start_time.isoformat(),
            "training_power_rule": "originally observed 15-minute bins only",
            "rf_inference_gap_policy": inference_gap_policy,
            "rf_inference_gap_rule": (
                "existing approved source fills only; require a complete causally available latest bin"
                if inference_gap_policy == "strict" else
                "RF inference only: also use plant-policy bounded test-gap linear fills (at most "
                "four minutes) after their closing originally observed minute is available; "
                "if only the latest bin's final minute is unavailable, hold the immediately "
                "preceding originally observed minute; no changes to training, calibration, "
                "observed counts, peak-run age, or original p90 histories"
            ),
            "rf_issue_input_rule": (
                "approved linearly filled bins may supply power features only after "
                "their closing originally observed minute is available; "
                "observed-count features and peak-run-age classification remain observed-only; "
                "the selected rf_inference_gap_rule specifies any inference-only "
                "plant fills and causal final-minute hold"
            ),
            "source_time_rule": "minute CSV timestamps are interval ends; 15-minute labels are interval starts",
            "source_to_simulation_mapping": "fixed local calendar offset with matching clock time",
            "forecast_interval_minutes": 15,
            "forecast_horizon_minutes": 480,
            "short_method": "rf_q75_peak",
            "short_leads_minutes": "15-120",
            "long_method": f"p90_{history_days}d",
            "long_leads_minutes": "135-480",
            "handoff_rule": "hard switch after the eighth 15-minute bin; no smoothing",
            "p90_rule": (
                f"clock-slot 90th percentile over {history_days} completed source "
                "calendar days, linear quantile interpolation; approved short-gap "
                "fills allowed except in the final hour before daily issue"
            ),
            "peak_threshold_w": threshold_w,
            "peak_threshold_rule": "95th percentile of fully observed 15-minute bins, August 2-21, 2026",
            "rf_fit": forest_details,
            "rf_q75_calibration": q75_details,
            "rf_peak_continuation": continuation_details,
            "source_details": source_details,
        }
        return cls(
            source_path=source_path,
            simulation_start=sim_start,
            source_start=source_start_time,
            history_days=history_days,
            approved=approved,
            observed=observed,
            approved_ready_at=approved_ready_at,
            forest=forest,
            q75_residual_w=np.asarray(q75_offsets, dtype=float),
            continuation=continuation,
            peak_threshold_w=threshold_w,
            metadata=metadata,
            inference_gap_policy=inference_gap_policy,
            _inference_bins=inference_bins,
            _inference_minutes=inference_minutes,
        )

    def source_timestamp(self, simulation_timestamp: str | pd.Timestamp) -> pd.Timestamp:
        """Map a simulation time to the source clock used by all model features."""
        simulation_wall = _wall_time(simulation_timestamp)
        if simulation_wall < self.simulation_start:
            raise ValueError("Rest-of-home forecast issue precedes the simulation replay start")
        if simulation_wall != simulation_wall.floor("15min"):
            raise ValueError("Rest-of-home forecast issue must align to a 15-minute boundary")
        source_time = self.source_start + (simulation_wall - self.simulation_start)
        if source_time not in self.observed.index:
            raise ValueError(f"Rest-of-home source issue time is absent: {source_time}")
        return source_time

    def forecast_kw(
        self, sim_timestamp: str | pd.Timestamp, steps: int = 32, *,
        rf_only: bool = False,
    ) -> pd.Series:
        """Return interval-start kW, using only measurements completed at issue.

        ``rf_only`` exposes the already computed conservative RF component at
        every lead for offline comparisons. The default two-hour RF to p90
        handoff remains the controller forecast.
        """
        if type(steps) is not int or not 1 <= steps <= STEPS:
            raise ValueError(f"steps must be an integer from 1 through {STEPS}")
        if type(rf_only) is not bool:
            raise ValueError("rf_only must be a boolean")
        sim_time = pd.Timestamp(sim_timestamp)
        source_time = self.source_timestamp(sim_time)
        latest_time = source_time - INTERVAL
        # An approved interpolation can serve as a power input only after the
        # observed right endpoint that completed its gap has arrived.
        visible = self.approved.loc[source_time - 8 * INTERVAL:source_time].copy()
        ready = self.approved_ready_at.reindex(visible.index)
        plant_fill_bins = pd.Series(False, index=visible.index)
        last_minute_hold = False
        if self.inference_gap_policy == "causal_short_gap":
            if self._inference_bins is None or self._inference_minutes is None:
                raise RuntimeError("Causal short-gap inference data was not initialized")
            extra = self._inference_bins.reindex(visible.index)
            plant_fill_bins = extra["plant_filled_minutes"].gt(0)
            visible.loc[plant_fill_bins] = extra.loc[plant_fill_bins, "power_w"]
            ready.loc[plant_fill_bins] = extra.loc[plant_fill_bins, "ready_at"]
        visible.loc[ready.isna() | ready.gt(source_time)] = np.nan
        visible.loc[source_time] = np.nan
        if self.inference_gap_policy == "causal_short_gap" and pd.isna(visible.loc[latest_time]):
            minute_index = pd.date_range(
                latest_time + pd.Timedelta(minutes=1), source_time, freq="min",
            )
            latest_minutes = self._inference_minutes.reindex(minute_index)
            available = latest_minutes["power_w"].where(
                latest_minutes["ready_at"].notna()
                & latest_minutes["ready_at"].le(source_time)
            )
            # Only a single unavailable final minute may use this causal hold.
            # In particular, never replace a multi-minute unresolved gap or
            # carry forward a preceding interpolated (not observed) reading.
            previous_observed = latest_minutes["observed_w"].iloc[-2]
            if (available.isna().sum() == 1 and pd.isna(available.iloc[-1])
                    and np.isfinite(previous_observed)):
                available.iloc[-1] = previous_observed
                visible.loc[latest_time] = float(available.mean())
                last_minute_hold = True
                plant_fill_bins.loc[latest_time] = bool(
                    latest_minutes["plant_filled"].iloc[:-1].any()
                )
        if latest_time not in visible.index or not np.isfinite(visible.loc[latest_time]):
            raise ValueError(
                f"Conservative RF needs the latest approved 15-minute bin "
                f"available by {source_time}; no future-dependent fill is allowed"
            )

        # Mask the target bin even though the source CSV may contain later data.
        # Power features can use ready approved fills. Observed-count features
        # still describe how many bins were originally measured, as at fit.
        feature = features_from_observed(visible).loc[source_time]
        observed_visible = self.observed.loc[visible.index].copy()
        observed_visible.loc[source_time] = np.nan
        observed_feature = features_from_observed(observed_visible).loc[source_time]
        for count_name in ("observed_count_1h", "observed_count_2h"):
            feature[count_name] = observed_feature[count_name]
        if not np.isfinite(feature.to_numpy(dtype=float)).all():
            raise ValueError(f"Conservative RF has incomplete issue features at {source_time}")
        rf_input = pd.DataFrame([feature.to_dict()] * steps)
        rf_input["lead_step"] = np.arange(1, steps + 1)
        rf_w = np.maximum(self.forest.predict(rf_input), 0.0)
        latest_w = float(feature["latest_w"])
        if latest_w >= self.peak_threshold_w:
            run_age, onset_known = observed_peak_run_age(
                self.observed, source_time, self.peak_threshold_w
            )
            curve_key = ("age_1" if run_age == 1 else "age_2plus") if onset_known else "pooled"
        else:
            curve_key = "pooled"
        near_w = conservative_peak_forecast(
            rf_w, self.q75_residual_w[:steps], latest_w,
            self.peak_threshold_w, self.continuation[curve_key][:steps],
        )
        forecast_w = np.asarray(near_w, dtype=float)
        if steps > SHORT_STEPS and not rf_only:
            day = source_time.normalize()
            if day not in self._daily_profiles:
                self._daily_profiles[day] = _p90_clock_profile(
                    self.approved, self.observed, source_time, self.history_days
                )
            target_source = pd.date_range(source_time, periods=steps, freq=INTERVAL)
            slots = target_source.hour * 4 + target_source.minute // 15
            forecast_w[SHORT_STEPS:] = self._daily_profiles[day].reindex(
                slots[SHORT_STEPS:]
            ).to_numpy(dtype=float)
        if forecast_w.shape != (steps,) or not np.isfinite(forecast_w).all() or (forecast_w < 0).any():
            raise ValueError("Rest-of-home hybrid forecast must contain finite nonnegative watts")
        simulation_targets = pd.date_range(sim_time, periods=steps, freq=INTERVAL)
        result = pd.Series(forecast_w / 1000.0, index=simulation_targets, name=POWER_NAME)
        available_fills = visible.iloc[:-1].notna() & self.observed.reindex(visible.index[:-1]).isna()
        result.attrs["rf_approved_fill_input_bins"] = int(available_fills.sum())
        result.attrs["rf_latest_input_approved_filled"] = bool(
            pd.isna(self.observed.loc[latest_time])
        )
        result.attrs["rf_inference_gap_policy"] = self.inference_gap_policy
        result.attrs["rf_plant_fill_input_bins"] = int((plant_fill_bins & visible.notna()).sum())
        result.attrs["rf_latest_bin_last_minute_hold"] = last_minute_hold
        result.attrs["rf_last_minute_hold_input_minutes"] = int(last_minute_hold)
        return result
