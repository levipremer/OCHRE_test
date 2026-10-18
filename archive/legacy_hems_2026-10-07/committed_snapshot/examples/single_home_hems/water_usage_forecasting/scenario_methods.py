"""Causal, whole-trajectory water-use scenarios from completed historical days.

The scenario generator deliberately has no fit step: each call sees only the
history available at its forecast origin. A scenario is a complete historical
future-shaped trajectory aligned by local wall clock, so draw timing and
duration within a candidate are kept together. Values are interval-average
flow in L/min and interval labels denote their starts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pytz.exceptions import AmbiguousTimeError, NonExistentTimeError


@dataclass(frozen=True)
class ScenarioConfig:
    horizon_hours: float = 12
    interval_minutes: int = 15
    lookback_days: int = 35
    candidate_policy: str = "daily"  # daily | same_weekday
    morning_policy: str = "none"  # none | remaining_budget | remaining_budget_high_only
    morning_start_hour: int = 6
    morning_end_hour: int = 12
    morning_budget_quantile: float = 0.5
    morning_progress_fraction: float = 0.75
    budget_scale_l: float = 10.0
    persistence_steps: int = 0
    persistence_strengths: tuple[float, ...] | None = None
    persistence_mode: str = "always"  # always | active_only
    active_threshold_l_per_min: float = 0.75

    def __post_init__(self) -> None:
        if isinstance(self.interval_minutes, bool) or not isinstance(self.interval_minutes, int):
            raise ValueError("interval_minutes must be a positive integer")
        if self.interval_minutes <= 0 or 1440 % self.interval_minutes:
            raise ValueError("interval_minutes must divide one 24-hour day")
        if not np.isfinite(self.horizon_hours) or self.horizon_hours <= 0:
            raise ValueError("horizon_hours must be positive and finite")
        steps = self.horizon_hours * 60 / self.interval_minutes
        if not np.isclose(steps, round(steps), rtol=0, atol=1e-9):
            raise ValueError("horizon_hours must contain a whole number of intervals")
        if isinstance(self.lookback_days, bool) or not isinstance(self.lookback_days, int) or self.lookback_days < 1:
            raise ValueError("lookback_days must be a positive integer")
        if self.candidate_policy not in {"daily", "same_weekday"}:
            raise ValueError("candidate_policy must be daily or same_weekday")
        if self.morning_policy not in {"none", "remaining_budget", "remaining_budget_high_only"}:
            raise ValueError("morning_policy must be none, remaining_budget, or remaining_budget_high_only")
        if not (isinstance(self.morning_start_hour, int) and isinstance(self.morning_end_hour, int)
                and 0 <= self.morning_start_hour < self.morning_end_hour <= 24):
            raise ValueError("morning hours must define a window within one local day")
        if not np.isfinite(self.morning_budget_quantile) or not 0 <= self.morning_budget_quantile <= 1:
            raise ValueError("morning_budget_quantile must lie between zero and one")
        if not np.isfinite(self.morning_progress_fraction) or not 0 <= self.morning_progress_fraction <= 1:
            raise ValueError("morning_progress_fraction must lie between zero and one")
        if not np.isfinite(self.budget_scale_l) or self.budget_scale_l <= 0:
            raise ValueError("budget_scale_l must be positive and finite")
        if isinstance(self.persistence_steps, bool) or not isinstance(self.persistence_steps, int):
            raise ValueError("persistence_steps must be a nonnegative integer")
        if not 0 <= self.persistence_steps <= round(steps):
            raise ValueError("persistence_steps must fit within the horizon")
        if self.persistence_strengths is not None:
            strengths = np.asarray(self.persistence_strengths, dtype=float)
            if strengths.ndim != 1 or len(strengths) != self.persistence_steps:
                raise ValueError("persistence_strengths must have one value per persistence step")
            if not np.isfinite(strengths).all() or ((strengths < 0) | (strengths > 1)).any():
                raise ValueError("persistence strengths must lie between zero and one")
        if self.persistence_mode not in {"always", "active_only"}:
            raise ValueError("persistence_mode must be always or active_only")
        if not np.isfinite(self.active_threshold_l_per_min) or self.active_threshold_l_per_min < 0:
            raise ValueError("active_threshold_l_per_min must be nonnegative and finite")


@dataclass(frozen=True)
class ScenarioSet:
    values: np.ndarray  # [historical candidate, future interval], L/min
    weights: np.ndarray  # positive and normalized
    candidate_ids: tuple[str, ...]  # local dates, newest first
    diagnostics: dict[str, object]


def apply_persistence(
    scenarios: ScenarioSet, history: pd.Series, config: ScenarioConfig
) -> ScenarioSet:
    """Blend a completed last value into the first requested scenario leads.

    A zero-step base ensemble may be reused for several persistence settings.
    Candidate trajectories and weights are otherwise unchanged. ``active_only``
    skips the blend unless the most recent finite observation meets the
    configured threshold.
    """
    if not isinstance(history.index, pd.DatetimeIndex) or history.index.tz is None:
        raise ValueError("history needs a timezone-aware DatetimeIndex")
    if history.index.has_duplicates or not history.index.is_monotonic_increasing:
        raise ValueError("history timestamps must be unique and increasing")
    values = np.asarray(scenarios.values, dtype=float)
    if values.ndim != 2 or values.shape[1] != round(config.horizon_hours * 60 / config.interval_minutes):
        raise ValueError("scenario horizon does not match persistence configuration")
    if config.persistence_steps and scenarios.diagnostics.get("persistence_steps_applied", 0):
        raise ValueError("Apply persistence to a base scenario set, not an already adapted set")
    origin = pd.Timestamp(scenarios.diagnostics["origin"])
    interval = pd.Timedelta(minutes=config.interval_minutes)
    # A past finite value can be stale after a missing observation. Only the
    # immediately preceding, fully completed interval can seed persistence.
    latest_start = origin - interval
    latest_value = history.reindex(pd.DatetimeIndex([latest_start])).iloc[0]
    latest = float(latest_value) if np.isfinite(latest_value) else None
    if latest is not None and latest < 0:
        raise ValueError("Completed water flow must be nonnegative")
    strengths = (
        np.ones(config.persistence_steps, dtype=float)
        if config.persistence_strengths is None
        else np.asarray(config.persistence_strengths, dtype=float)
    )
    active = latest is not None and config.persistence_steps > 0 and (
        config.persistence_mode == "always" or latest >= config.active_threshold_l_per_min
    )
    diagnostics = dict(scenarios.diagnostics)
    diagnostics.update({
        "persistence_steps_requested": config.persistence_steps,
        "persistence_steps_applied": config.persistence_steps if active else 0,
        "persistence_strengths_requested": strengths.tolist(),
        "persistence_strengths_applied": strengths.tolist() if active else [],
        "persistence_mode": config.persistence_mode,
        "persistence_last_flow_l_per_min": latest,
        "persistence_last_bin_observed": latest is not None,
        "persistence_active_threshold_l_per_min": config.active_threshold_l_per_min,
        "persistence_active": active,
        "persistence_skip_reason": (
            "missing_latest_completed_bin" if latest is None else
            "below_active_threshold" if config.persistence_steps and not active else
            "no_steps_requested" if not config.persistence_steps else None
        ),
    })
    if not active:
        return ScenarioSet(values, scenarios.weights, scenarios.candidate_ids, diagnostics)
    adapted = values.copy()
    adapted[:, :config.persistence_steps] = (
        adapted[:, :config.persistence_steps] * (1 - strengths)
        + latest * strengths
    )
    return ScenarioSet(adapted, scenarios.weights, scenarios.candidate_ids, diagnostics)


def _local_window(day: pd.Timestamp, start_hour: int, end_hour: int, timezone: object) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return unique morning bounds; callers skip days with unusual clock rules."""
    start = (day + pd.Timedelta(hours=start_hour)).tz_localize(
        timezone, ambiguous="raise", nonexistent="raise"
    )
    end = (day + pd.Timedelta(hours=end_hour)).tz_localize(
        timezone, ambiguous="raise", nonexistent="raise"
    )
    return start, end


def _observed_volume(
    history: pd.Series, start: pd.Timestamp, end: pd.Timestamp, interval: pd.Timedelta
) -> float | None:
    """Integrate completed interval means; return None for any coverage gap."""
    if end <= start:
        return 0.0
    index = history.index
    selected = history.loc[(index < end) & (index + interval > start)]
    cursor = start
    volume = 0.0
    for bin_start, flow in selected.items():
        part_start = max(bin_start, start)
        part_end = min(bin_start + interval, end)
        if part_end <= part_start:
            continue
        if part_start != cursor or not np.isfinite(flow):
            return None
        volume += float(flow) * (part_end - part_start).total_seconds() / 60
        cursor = part_end
    return volume if cursor == end else None


def _candidate_remaining_volumes(
    values: np.ndarray, future: pd.DatetimeIndex, end: pd.Timestamp, interval: pd.Timedelta
) -> np.ndarray:
    overlap_minutes = np.asarray([
        max(0.0, (min(start + interval, end) - start).total_seconds() / 60)
        for start in future
    ])
    return values @ overlap_minutes


def generate_scenarios(
    history: pd.Series,
    origin: pd.Timestamp,
    future: pd.DatetimeIndex,
    config: ScenarioConfig,
) -> ScenarioSet:
    """Build and condition complete historical 12-hour analogs causally.

    ``future`` starts at ``origin`` and has the configured duration. Completed
    observations at or after the origin are ignored, even if a caller passes
    them by mistake. Ambiguous/nonexistent historical wall-clock paths and
    paths with gaps or missing readings are excluded rather than interpolated.
    """
    if not isinstance(history.index, pd.DatetimeIndex) or history.index.tz is None:
        raise ValueError("history needs a timezone-aware DatetimeIndex")
    if history.index.has_duplicates or not history.index.is_monotonic_increasing:
        raise ValueError("history timestamps must be unique and increasing")
    if not isinstance(future, pd.DatetimeIndex) or future.tz is None:
        raise ValueError("future needs a timezone-aware DatetimeIndex")
    origin = pd.Timestamp(origin)
    if origin.tz is None:
        raise ValueError("origin must be timezone-aware")
    timezone = history.index.tz
    origin = origin.tz_convert(timezone)
    future = future.tz_convert(timezone)
    interval = pd.Timedelta(minutes=config.interval_minutes)
    steps = round(config.horizon_hours * 60 / config.interval_minutes)
    expected = pd.date_range(origin, periods=steps, freq=interval)
    if not future.equals(expected):
        raise ValueError("future must be the complete regular grid starting at origin")

    # Interval-start labels represent observations available only at interval end.
    completed = history.loc[history.index + interval <= origin].astype(float)
    ignored_future_rows = len(history) - len(completed)
    if completed.empty or not np.isfinite(completed.to_numpy()).any():
        raise ValueError("No finite completed history is available")
    if (completed.dropna() < 0).any() or np.isinf(completed.to_numpy()).any():
        raise ValueError("Completed water flow must be nonnegative and finite or missing")

    future_wall = future.tz_localize(None)

    def collect(policy: str) -> tuple[list[np.ndarray], list[str], dict[str, int]]:
        trajectories: list[np.ndarray] = []
        candidate_ids: list[str] = []
        exclusions = {"weekday_mismatch": 0, "clock_transition": 0,
                      "not_completed": 0, "missing_observations": 0}
        for days_back in range(1, config.lookback_days + 1):
            if policy == "same_weekday" and days_back % 7:
                exclusions["weekday_mismatch"] += 1
                continue
            # Shift *wall-clock* labels, not elapsed UTC time. A historical
            # DST transition is used only when the entire path is unambiguous
            # and has the same uninterrupted elapsed grid as today's horizon.
            candidate_wall = future_wall - pd.Timedelta(days=days_back)
            try:
                candidate = candidate_wall.tz_localize(
                    timezone, ambiguous="raise", nonexistent="raise"
                )
            except (ValueError, TypeError, AmbiguousTimeError, NonExistentTimeError):
                exclusions["clock_transition"] += 1
                continue
            if candidate.has_duplicates or not candidate.equals(
                pd.date_range(candidate[0], periods=steps, freq=interval)
            ):
                exclusions["clock_transition"] += 1
                continue
            if candidate[-1] + interval > origin:
                exclusions["not_completed"] += 1
                continue
            trajectory = completed.reindex(candidate).to_numpy(dtype=float)
            if not np.isfinite(trajectory).all():
                exclusions["missing_observations"] += 1
                continue
            trajectories.append(trajectory)
            candidate_ids.append(candidate[0].date().isoformat())
        return trajectories, candidate_ids, exclusions

    trajectories, candidate_ids, exclusions = collect(config.candidate_policy)
    effective_policy = config.candidate_policy
    fallback_exclusions: dict[str, int] | None = None
    if not trajectories and config.candidate_policy == "same_weekday":
        trajectories, candidate_ids, fallback_exclusions = collect("daily")
        effective_policy = "daily"
    if not trajectories:
        raise ValueError("No complete historical scenario exists within the lookback window")

    values = np.stack(trajectories).astype(float, copy=False)
    weights = np.full(len(values), 1 / len(values), dtype=float)
    diagnostics: dict[str, object] = {
        "candidate_count": len(values),
        "origin": origin.isoformat(),
        "candidate_days_examined": config.lookback_days,
        "requested_candidate_policy": config.candidate_policy,
        "effective_candidate_policy": effective_policy,
        "same_weekday_fallback_to_daily": effective_policy != config.candidate_policy,
        "candidate_exclusion_counts": exclusions,
        "fallback_exclusion_counts": fallback_exclusions,
        "history_future_rows_ignored": ignored_future_rows,
        "morning_condition_active": False,
        "morning_condition_reason": "disabled",
        "current_morning_volume_l": None,
        "typical_morning_volume_l": None,
        "remaining_budget_l": None,
        "morning_progress_fraction": config.morning_progress_fraction,
        "morning_progress_ratio": None,
        "morning_progress_threshold_l": None,
        "morning_progress_gate_passed": None,
        "morning_reference_days": 0,
        "candidate_remaining_morning_l": None,
        "persistence_steps_requested": config.persistence_steps,
        "persistence_steps_applied": 0,
        "persistence_strengths_requested": [],
        "persistence_strengths_applied": [],
        "persistence_mode": config.persistence_mode,
        "persistence_last_flow_l_per_min": None,
        "persistence_active_threshold_l_per_min": config.active_threshold_l_per_min,
        "persistence_active": False,
    }

    if config.morning_policy in {"remaining_budget", "remaining_budget_high_only"}:
        local_day = pd.Timestamp(origin.date())
        try:
            morning_start, morning_end = _local_window(
                local_day, config.morning_start_hour, config.morning_end_hour, timezone
            )
        except (ValueError, TypeError, AmbiguousTimeError, NonExistentTimeError):
            diagnostics["morning_condition_reason"] = "ambiguous_morning_clock"
        else:
            if not morning_start <= origin < morning_end:
                diagnostics["morning_condition_reason"] = "origin_outside_morning"
            else:
                observed = _observed_volume(completed, morning_start, origin, interval)
                if observed is None:
                    diagnostics["morning_condition_reason"] = "incomplete_current_morning"
                else:
                    reference: list[float] = []
                    for days_back in range(1, config.lookback_days + 1):
                        if effective_policy == "same_weekday" and days_back % 7:
                            continue
                        day = local_day - pd.Timedelta(days=days_back)
                        try:
                            start, end = _local_window(
                                day, config.morning_start_hour, config.morning_end_hour, timezone
                            )
                        except (ValueError, TypeError, AmbiguousTimeError, NonExistentTimeError):
                            continue
                        if end <= origin:
                            total = _observed_volume(completed, start, end, interval)
                            if total is not None:
                                reference.append(total)
                    diagnostics["current_morning_volume_l"] = observed
                    diagnostics["morning_reference_days"] = len(reference)
                    if not reference:
                        diagnostics["morning_condition_reason"] = "no_complete_reference_mornings"
                    else:
                        typical = float(np.quantile(reference, config.morning_budget_quantile))
                        remaining_budget = max(typical - observed, 0.0)
                        remaining = _candidate_remaining_volumes(values, future, morning_end, interval)
                        threshold = config.morning_progress_fraction * typical
                        gate_passed = typical > 0 and observed >= threshold
                        diagnostics.update({
                            "typical_morning_volume_l": typical,
                            "remaining_budget_l": remaining_budget,
                            "candidate_remaining_morning_l": remaining.tolist(),
                            "morning_progress_ratio": observed / typical if typical > 0 else None,
                            "morning_progress_threshold_l": threshold,
                            "morning_progress_gate_passed": gate_passed,
                        })
                        if config.morning_policy == "remaining_budget_high_only" and not gate_passed:
                            diagnostics["morning_condition_reason"] = (
                                "zero_typical_morning_volume" if typical == 0 else
                                "below_progress_threshold"
                            )
                        else:
                            penalty = np.maximum(remaining - remaining_budget, 0.0) / config.budget_scale_l
                            # Keep five percent uniform weight: even an unusually
                            # large second draw stays possible, not forbidden.
                            relative = np.exp(-(penalty - np.min(penalty)))
                            tilted = relative / relative.sum()
                            weights = 0.95 * tilted + 0.05 / len(values)
                            weights /= weights.sum()
                            diagnostics["morning_condition_active"] = True
                            diagnostics["morning_condition_reason"] = "applied"

    diagnostics["effective_scenario_count"] = float(1 / np.sum(weights**2))
    diagnostics["max_scenario_weight"] = float(np.max(weights))

    return apply_persistence(
        ScenarioSet(values, weights, tuple(candidate_ids), diagnostics), history, config
    )
