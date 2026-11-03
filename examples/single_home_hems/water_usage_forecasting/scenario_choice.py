"""Equal-weight selection of complete historical water-use paths.

Every rule ranks the same eligible, same-local-clock daily path pool. A path
is never stitched or edited: only its place in the ranking changes. The flow
series uses interval-start labels, so observations are available only after
their intervals end. All selection features are computed before the origin.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date

import numpy as np
import pandas as pd
from pytz.exceptions import AmbiguousTimeError, NonExistentTimeError

from .scenario_methods import ScenarioConfig, ScenarioSet, generate_scenarios


@dataclass(frozen=True)
class ChoiceConfig:
    method: str  # recent | rmse | rmse_volume | day_type | weekday | progress | tail_anchored
    count: int
    hours_preceding: int = 24
    volume_weight: float = 0.5
    progress_start_hour: int | None = None
    progress_end_hour: int | None = None
    progress_block_hours: int | None = None

    def __post_init__(self) -> None:
        if self.method not in {"recent", "rmse", "rmse_volume", "day_type", "weekday", "progress", "tail_anchored"}:
            raise ValueError("Unknown scenario choice method")
        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count < 1:
            raise ValueError("count must be a positive integer")
        if (isinstance(self.hours_preceding, bool)
                or not isinstance(self.hours_preceding, int) or self.hours_preceding < 1):
            raise ValueError("hours_preceding must be a positive integer")
        if not np.isfinite(self.volume_weight) or not 0 <= self.volume_weight <= 1:
            raise ValueError("volume_weight must lie between zero and one")
        if self.method == "tail_anchored":
            if self.count < 2:
                raise ValueError("tail_anchored needs at least two paths")
        specified_window = self.progress_start_hour is not None or self.progress_end_hour is not None
        if self.method == "progress":
            if specified_window == (self.progress_block_hours is not None):
                raise ValueError("progress needs either a local window or fixed local blocks")
            if specified_window:
                start, end = self.progress_start_hour, self.progress_end_hour
                if (isinstance(start, bool) or isinstance(end, bool)
                        or not isinstance(start, int) or not isinstance(end, int)
                        or not 0 <= start < end <= 24):
                    raise ValueError("progress window must be ordered local hours in [0, 24]")
            elif self.progress_block_hours not in {6, 12}:
                raise ValueError("progress_block_hours must be 6 or 12")
        elif specified_window or self.progress_block_hours is not None:
            raise ValueError("Progress window settings require method='progress'")


def _completed(history: pd.Series, origin: pd.Timestamp, interval: pd.Timedelta) -> pd.Series:
    if not isinstance(history.index, pd.DatetimeIndex) or history.index.tz is None:
        raise ValueError("history needs a timezone-aware DatetimeIndex")
    if history.index.has_duplicates or not history.index.is_monotonic_increasing:
        raise ValueError("history timestamps must be unique and increasing")
    completed = history.loc[history.index + interval <= origin].astype(float)
    if np.isinf(completed.to_numpy()).any() or (completed.dropna() < 0).any():
        raise ValueError("Completed water flow must be nonnegative and finite or missing")
    return completed


def _candidate_offset(origin: pd.Timestamp, candidate_id: str) -> int:
    return (origin.date() - date.fromisoformat(candidate_id)).days


def _localize_wall(index: pd.DatetimeIndex, timezone: object) -> pd.DatetimeIndex | None:
    try:
        return index.tz_localize(timezone, ambiguous="raise", nonexistent="raise")
    except (ValueError, TypeError, AmbiguousTimeError, NonExistentTimeError):
        return None


def _past_match_scores(
    completed: pd.Series,
    origin: pd.Timestamp,
    future: pd.DatetimeIndex,
    candidate_ids: tuple[str, ...],
    interval: pd.Timedelta,
    choice: ChoiceConfig,
) -> tuple[list[tuple[int, float]], dict[str, int], dict[int, tuple[float, float]]]:
    past_steps, remainder = divmod(choice.hours_preceding * 60, int(interval.total_seconds() / 60))
    if remainder:
        raise ValueError("hours_preceding must contain a whole number of intervals")
    current_past = pd.date_range(origin - past_steps * interval, periods=past_steps, freq=interval)
    current = completed.reindex(current_past).to_numpy(dtype=float)
    if not np.isfinite(current).all():
        raise ValueError("The current preceding window has missing completed readings")

    target_wall = current_past.append(future).tz_localize(None)
    ranked: list[tuple[int, float]] = []
    details: dict[int, tuple[float, float]] = {}
    exclusions = {"clock_transition": 0, "not_completed": 0, "missing_past": 0}
    for index, candidate_id in enumerate(candidate_ids):
        days_back = _candidate_offset(origin, candidate_id)
        candidate = _localize_wall(target_wall - pd.Timedelta(days=days_back), origin.tz)
        if (candidate is None or candidate.has_duplicates or not candidate.equals(
                pd.date_range(candidate[0], periods=len(candidate), freq=interval))):
            exclusions["clock_transition"] += 1
            continue
        if candidate[-1] + interval > origin:
            exclusions["not_completed"] += 1
            continue
        previous = completed.reindex(candidate[:past_steps]).to_numpy(dtype=float)
        if not np.isfinite(previous).all():
            exclusions["missing_past"] += 1
            continue
        rmse = float(np.sqrt(np.mean((previous - current) ** 2)))
        volume_difference_l = float(abs(np.sum(previous - current)) * interval.total_seconds() / 60)
        volume_rate_difference = volume_difference_l / (choice.hours_preceding * 60)
        score = (rmse if choice.method == "rmse" else
                 (1 - choice.volume_weight) * rmse + choice.volume_weight * volume_rate_difference)
        ranked.append((index, score))
        details[index] = (rmse, volume_difference_l)
    return ranked, exclusions, details


def _progress_start(origin: pd.Timestamp, choice: ChoiceConfig) -> tuple[pd.Timestamp | None, str | None]:
    hour = origin.hour + origin.minute / 60 + origin.second / 3600
    if choice.progress_block_hours is not None:
        start_hour = (origin.hour // choice.progress_block_hours) * choice.progress_block_hours
    else:
        assert choice.progress_start_hour is not None and choice.progress_end_hour is not None
        if not choice.progress_start_hour <= hour < choice.progress_end_hour:
            return None, "outside_progress_window"
        start_hour = choice.progress_start_hour
    if hour == start_hour:
        return None, "at_progress_window_start"
    start_wall = pd.Timestamp(origin.date()) + pd.Timedelta(hours=start_hour)
    try:
        start = start_wall.tz_localize(origin.tz, ambiguous="raise", nonexistent="raise")
    except (ValueError, TypeError, AmbiguousTimeError, NonExistentTimeError):
        return None, "ambiguous_progress_window"
    return start, None


def _volume_between(
    completed: pd.Series, start: pd.Timestamp, end: pd.Timestamp, interval: pd.Timedelta,
) -> float | None:
    if end <= start:
        return 0.0 if end == start else None
    expected = pd.date_range(start, end, inclusive="left", freq=interval)
    if len(expected) == 0 or expected[-1] + interval != end:
        return None
    values = completed.reindex(expected).to_numpy(dtype=float)
    if not np.isfinite(values).all():
        return None
    return float(values.sum() * interval.total_seconds() / 60)


def _progress_scores(
    completed: pd.Series,
    origin: pd.Timestamp,
    candidate_ids: tuple[str, ...],
    interval: pd.Timedelta,
    choice: ChoiceConfig,
) -> tuple[list[tuple[int, float]], dict[str, int], str | None, float | None]:
    start, fallback = _progress_start(origin, choice)
    if start is None:
        return [(i, 0.0) for i in range(len(candidate_ids))], {}, fallback, None
    # A fixed local-clock block can contain a different number of elapsed
    # intervals on a daylight-saving transition day. Do not compare its
    # cumulative volume with a normal-length block.
    current_elapsed = origin - start
    current_wall = origin.tz_localize(None) - start.tz_localize(None)
    if current_elapsed != current_wall:
        return [(i, 0.0) for i in range(len(candidate_ids))], {}, "current_clock_transition", None
    current_volume = _volume_between(completed, start, origin, interval)
    if current_volume is None:
        return [(i, 0.0) for i in range(len(candidate_ids))], {}, "missing_current_progress", None
    ranked = []
    exclusions = {"ambiguous_candidate_clock": 0, "clock_transition_progress_duration": 0,
                  "missing_candidate_progress": 0}
    start_wall = pd.DatetimeIndex([start]).tz_localize(None)
    origin_wall = pd.DatetimeIndex([origin]).tz_localize(None)
    for index, candidate_id in enumerate(candidate_ids):
        days_back = _candidate_offset(origin, candidate_id)
        candidate_bounds = _localize_wall(
            start_wall.append(origin_wall) - pd.Timedelta(days=days_back), origin.tz,
        )
        if candidate_bounds is None:
            exclusions["ambiguous_candidate_clock"] += 1
            continue
        candidate_start, candidate_end = candidate_bounds
        if candidate_end > origin:
            exclusions["ambiguous_candidate_clock"] += 1
            continue
        if candidate_end - candidate_start != current_elapsed:
            exclusions["clock_transition_progress_duration"] += 1
            continue
        candidate_volume = _volume_between(completed, candidate_start, candidate_end, interval)
        if candidate_volume is None:
            exclusions["missing_candidate_progress"] += 1
            continue
        ranked.append((index, abs(candidate_volume - current_volume)))
    return ranked, exclusions, None, current_volume


def select_scenarios(
    history: pd.Series,
    origin: pd.Timestamp,
    future: pd.DatetimeIndex,
    base_config: ScenarioConfig,
    choice_config: ChoiceConfig,
    *,
    candidate_pool: ScenarioSet | None = None,
) -> ScenarioSet:
    """Rank distinct complete daily paths and select up to ``count`` equally.

    ``candidate_pool`` can be a previously generated daily, unconditioned pool
    for this origin. Supplying it once per origin avoids reconstructing the
    candidate future paths for every choice method and K.
    """
    origin = pd.Timestamp(origin)
    if origin.tz is None:
        raise ValueError("origin must be timezone-aware")
    if not isinstance(history.index, pd.DatetimeIndex) or history.index.tz is None:
        raise ValueError("history needs a timezone-aware DatetimeIndex")
    if not isinstance(future, pd.DatetimeIndex) or future.tz is None:
        raise ValueError("future needs a timezone-aware DatetimeIndex")
    interval = pd.Timedelta(minutes=base_config.interval_minutes)
    origin = origin.tz_convert(history.index.tz)
    future = future.tz_convert(history.index.tz)
    completed = _completed(history, origin, interval)
    if candidate_pool is None:
        base = replace(base_config, candidate_policy="daily", morning_policy="none",
                       persistence_steps=0, persistence_strengths=None)
        candidate_pool = generate_scenarios(history, origin, future, base)
    expected_steps = round(base_config.horizon_hours * 60 / base_config.interval_minutes)
    if (candidate_pool.diagnostics.get("origin") != origin.isoformat()
            or candidate_pool.diagnostics.get("requested_candidate_policy") != "daily"
            or candidate_pool.diagnostics.get("selection_method") is not None
            or candidate_pool.diagnostics.get("morning_condition_active")
            or candidate_pool.diagnostics.get("persistence_steps_applied")
            or len(candidate_pool.candidate_ids) == 0
            or candidate_pool.values.shape != (len(candidate_pool.candidate_ids), expected_steps)
            or len(set(candidate_pool.candidate_ids)) != len(candidate_pool.candidate_ids)
            or not np.isfinite(candidate_pool.values).all()
            or candidate_pool.weights.shape != (len(candidate_pool.candidate_ids),)
            or not np.allclose(candidate_pool.weights,
                               np.full(len(candidate_pool.candidate_ids),
                                       1 / len(candidate_pool.candidate_ids)),
                               rtol=0, atol=1e-12)):
        raise ValueError("candidate_pool must contain this origin's distinct, unconditioned daily paths")
    if not future.equals(pd.date_range(origin, periods=expected_steps, freq=interval)):
        raise ValueError("future must be the complete regular grid starting at origin")

    ids = candidate_pool.candidate_ids
    exclusions: dict[str, int] = {}
    details: dict[int, tuple[float, float]] = {}
    fallback: str | None = None
    current_progress: float | None = None
    score_unit: str | None = None
    tail_diagnostics: dict[str, object] = {}
    if choice_config.method == "recent":
        ranked = [(i, 0.0) for i in range(len(ids))]
    elif choice_config.method in {"day_type", "weekday"}:
        current_weekday = origin.weekday()
        current_weekend = current_weekday >= 5
        ranked = [
            (i, float(
                (date.fromisoformat(candidate_id).weekday() >= 5) != current_weekend
                if choice_config.method == "day_type" else
                date.fromisoformat(candidate_id).weekday() != current_weekday
            ))
            for i, candidate_id in enumerate(ids)
        ]
        # Backfill from the freshest other eligible days after preferred dates.
        ranked.sort(key=lambda item: (item[1], item[0]))
    elif choice_config.method in {"rmse", "rmse_volume", "tail_anchored"}:
        ranked, exclusions, details = _past_match_scores(
            completed, origin, future, ids, interval, choice_config,
        )
        ranked.sort(key=lambda item: (item[1], item[0]))
        score_unit = "L/min"
        if choice_config.method == "tail_anchored":
            volumes = {
                index: float(candidate_pool.values[index].sum() * interval.total_seconds() / 60)
                for index, _ in ranked
            }
            if ranked:
                low = min(ranked, key=lambda item: (volumes[item[0]], item[0]))[0]
                high = min(ranked, key=lambda item: (-volumes[item[0]], item[0]))[0]
                selected = ranked[:max(0, choice_config.count - 2)]
                selected_ids = {index for index, _ in selected}
                for anchor in (low, high):
                    if anchor not in selected_ids:
                        selected.append((anchor, dict(ranked)[anchor]))
                        selected_ids.add(anchor)
                for item in ranked:
                    if len(selected) >= choice_config.count:
                        break
                    if item[0] not in selected_ids:
                        selected.append(item)
                        selected_ids.add(item[0])
                tail_diagnostics = {
                    "tail_ranking_hours": float(base_config.horizon_hours),
                    "tail_ranking_policy": "forecast_horizon",
                    "tail_low_candidate_id": ids[low],
                    "tail_high_candidate_id": ids[high],
                    "tail_low_volume_l": volumes[low],
                    "tail_high_volume_l": volumes[high],
                    "selected_tail_volumes_l": [volumes[index] for index, _ in selected],
                }
    else:
        ranked, exclusions, fallback, current_progress = _progress_scores(
            completed, origin, ids, interval, choice_config,
        )
        ranked.sort(key=lambda item: (item[1], item[0]))
        score_unit = "L" if fallback is None else None
    if not ranked:
        raise ValueError("No complete historical path has the required selection observations")

    if choice_config.method != "tail_anchored":
        selected = ranked[:choice_config.count]
    indices = [index for index, _ in selected]
    count = len(indices)
    backfill_count = (sum(score > 0 for _, score in selected)
                      if choice_config.method in {"day_type", "weekday"} else 0)
    if backfill_count:
        fallback = "calendar_recency_backfill"
    weights = np.full(count, 1 / count, dtype=float)
    diagnostics = dict(candidate_pool.diagnostics)
    diagnostics.update({
        "candidate_count": count,
        "requested_candidate_count": choice_config.count,
        "eligible_candidate_count": len(ranked),
        "selected_candidate_count": count,
        "selection_pool_count": len(ids),
        "selection_method": choice_config.method,
        "selection_fallback_reason": fallback,
        "calendar_recency_backfill_count": backfill_count,
        "selection_exclusion_counts": exclusions,
        "selected_match_scores": [score for _, score in selected],
        "selected_score_unit": score_unit,
        "selected_rmse_l_per_min": [details[i][0] for i in indices] if details else None,
        "selected_volume_difference_l": [details[i][1] for i in indices] if details else None,
        "hours_preceding": choice_config.hours_preceding if details else None,
        "volume_weight": choice_config.volume_weight if choice_config.method in {"rmse_volume", "tail_anchored"} else None,
        "progress_start_hour": choice_config.progress_start_hour,
        "progress_end_hour": choice_config.progress_end_hour,
        "progress_block_hours": choice_config.progress_block_hours,
        "current_progress_volume_l": current_progress,
        "scenario_weighting": "equal",
        "effective_scenario_count": float(count),
        "max_scenario_weight": float(1 / count),
        **tail_diagnostics,
    })
    return ScenarioSet(candidate_pool.values[indices].copy(), weights,
                       tuple(ids[index] for index in indices), diagnostics)


def truncate_choice_scenarios(scenarios: ScenarioSet, count: int) -> ScenarioSet:
    """Reuse one K-independent ranking for smaller, equally weighted choices."""
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")
    if scenarios.diagnostics.get("selection_method") is None:
        raise ValueError("scenarios must come from select_scenarios")
    if scenarios.diagnostics["selection_method"] == "tail_anchored" and count != len(scenarios.values):
        raise ValueError("tail_anchored must be reranked for each requested count")
    n = min(count, len(scenarios.values))
    diagnostics = dict(scenarios.diagnostics)
    backfill_count = (sum(score > 0 for score in diagnostics["selected_match_scores"][:n])
                      if diagnostics["selection_method"] in {"day_type", "weekday"} else 0)
    fallback = diagnostics["selection_fallback_reason"]
    if diagnostics["selection_method"] in {"day_type", "weekday"}:
        fallback = "calendar_recency_backfill" if backfill_count else None
    diagnostics.update({
        "candidate_count": n,
        "requested_candidate_count": count,
        "selected_candidate_count": n,
        "calendar_recency_backfill_count": backfill_count,
        "selection_fallback_reason": fallback,
        "selected_match_scores": diagnostics["selected_match_scores"][:n],
        "selected_rmse_l_per_min": (
            diagnostics["selected_rmse_l_per_min"][:n]
            if diagnostics["selected_rmse_l_per_min"] is not None else None
        ),
        "selected_volume_difference_l": (
            diagnostics["selected_volume_difference_l"][:n]
            if diagnostics["selected_volume_difference_l"] is not None else None
        ),
        "effective_scenario_count": float(n),
        "max_scenario_weight": float(1 / n),
    })
    return ScenarioSet(scenarios.values[:n].copy(), np.full(n, 1 / n),
                       scenarios.candidate_ids[:n], diagnostics)
