"""Causal historical EV analogs with a state-persistence forecast prefix.

Selection mirrors scenario-based-EV-prediction-analysis, with completed-path
eligibility and non-overlap added. Historical paths retain their supplied trip
SOC depletion when that field is available.
Prices and the selected backed-load forecast are shared across mobility paths;
paired Forecast objects retain their issued load provenance.
"""
from dataclasses import replace
import math
import numpy as np
import pandas as pd

from examples.five_home_block.five_home_block.ev_schedule import EVSchedule
from .control_forecasts import EVScenarioSet


FILTERS = {
    "tod_only": (("tod",),),
    "tod_dow": (("tod", "dow"),),
    "tod_state": (("tod", "state"),),
    "tod_dow_state": (("tod", "dow", "state"),),
    "tod_dow_state_hier": (("tod", "dow", "state"), ("tod", "dow"), ("tod",)),
    "tod_dow_state_hier_limit_persistence": (("tod", "dow", "state"), ("tod", "dow"), ("tod",)),
    "tod_dow_hier": (("tod", "dow"), ("tod",)),
    "tod_state_hier": (("tod", "state"), ("tod",)),
}
SCENARIO_DEFAULTS = dict(scenario_method="tod_dow_state_hier_limit_persistence", scenario_count=5,
    lookback_days=30, knn_history_hours=24, scenario_minutes=15, prefix_control_intervals=4)


class NoEVScenarios(ValueError):
    """No valid historical forecast: controller must use its baseline fallback."""


def _round_instant(timestamp: pd.Timestamp, grid: pd.Timedelta, direction: str) -> pd.Timestamp:
    """Round an absolute instant while retaining its local DST fold."""
    timestamp = pd.Timestamp(timestamp)
    clock = timestamp.tz_convert("UTC") if timestamp.tz is not None else timestamp
    rounded = getattr(clock, direction)(grid)
    return rounded.tz_convert(timestamp.tz) if timestamp.tz is not None else rounded


def scenario_settings(forecast, control_minutes, horizon_hours):
    """Resolve historical sampling defaults and validate grids and prefix length."""
    result = {k: forecast.get(k, v) for k, v in SCENARIO_DEFAULTS.items()}
    if result["scenario_method"] not in {*FILTERS, "knn_state"}:
        raise ValueError("Unknown EV scenario_method")
    for key in ("scenario_count", "scenario_minutes", "prefix_control_intervals"):
        v = result[key]
        if isinstance(v, bool) or not isinstance(v, int) or v < (0 if key.startswith("prefix") else 1):
            raise ValueError(f"EV {key} must be a {'nonnegative' if key.startswith('prefix') else 'positive'} integer")
    for key in ("lookback_days", "knn_history_hours"):
        v = result[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            raise ValueError(f"EV {key} must be positive and finite")
    minutes = result["scenario_minutes"]
    if 1440 % minutes or result["knn_history_hours"]*60 % minutes:
        raise ValueError("EV scenario grid must divide a day and the kNN history")
    if result["prefix_control_intervals"]*control_minutes > horizon_hours*60:
        raise ValueError("EV persistence prefix must fit within the prediction horizon")
    return result


def last_observed_arrival(mobility, timestamp):
    """Return the latest realized return known at issue, including that instant.

    Issued paths omit completed trips, so this observation must come from the
    realized schedule before selecting or shifting any historical scenario.
    """
    completed = mobility.trips.loc[mobility.trips.arrival <= timestamp, "arrival"]
    return None if completed.empty else str(completed.max())


def persistent_prefix(mobility, timestamp, horizon, duration, available, plant_step):
    """Overwrite availability and attach suppressed demand to the next trip.

    Exact boundary arrivals are delayed too (next arrival must be AFTER prefix).
    If no such arrival is in this horizon, retain deferred-demand diagnostics.
    The measured state always anchors the first plant interval, even at prefix=0.
    """
    end = timestamp + horizon
    boundary = timestamp + max(duration, plant_step)
    trips = mobility.trips.copy()
    overwritten = trips.loc[(trips.arrival > timestamp) & (trips.arrival <= boundary)]
    pending = float(overwritten.distance_km.sum())
    has_soc_depletion = "arrival_net_soc_depletion_pp" in trips
    pending_soc = (float(overwritten.arrival_net_soc_depletion_pp.sum())
                   if has_soc_depletion else None)
    retained = trips.loc[trips.arrival > boundary].copy()
    retained = retained.loc[retained.departure < end].reset_index(drop=True)
    deferred = pending
    deferred_soc = pending_soc
    if not retained.empty:
        if available:
            retained.loc[0, "departure"] = max(boundary, retained.loc[0, "departure"])
        else:
            # Already away, not a newly predicted departure at the solve instant.
            retained.loc[0, "departure"] = timestamp - plant_step
        retained.loc[0, "distance_km"] += pending
        if has_soc_depletion:
            retained.loc[0, "arrival_net_soc_depletion_pp"] += pending_soc
        if retained.loc[0, "arrival"] <= end:
            deferred = 0.0
            if has_soc_depletion:
                deferred_soc = 0.0
    elif not available:
        row = dict(departure=timestamp-plant_step,
                   arrival=end+plant_step, distance_km=pending)
        if has_soc_depletion:
            row["arrival_net_soc_depletion_pp"] = pending_soc
        retained = pd.DataFrame([row])
    metadata = dict(mobility.metadata, prefix_end=str(boundary),
        prefix_state="home" if available else "away", overwritten_distance_km=pending,
        deferred_distance_km=deferred)
    if has_soc_depletion:
        metadata.update(overwritten_soc_depletion_pp=pending_soc,
                        deferred_soc_depletion_pp=deferred_soc)
    return EVSchedule(retained, timestamp, end+plant_step, metadata)


class ScenarioEVForecast:
    """Per-EV forecast issuance; no future realized mobility may enter selection."""
    def __init__(self, vehicle_id, control):
        self.vehicle_id = vehicle_id
        self.control_minutes = control["control_minutes"]
        self.forecast_length = pd.Timedelta(hours=control.get("forecast_hours", control["horizon_hours"]))
        self.settings = scenario_settings(control["forecast"], self.control_minutes, control["horizon_hours"])
        self.history = []
        # Store final forecasts, not future truth. Scoring happens after simulation.
        self.issued_forecasts = []

    def at(self, timestamp, realized, horizon):
        timestamp = pd.Timestamp(timestamp)
        hub_intervals = realized.ev_hub_intervals
        if hub_intervals is not None and realized.backed_load_kw is not None:
            raise ValueError("EV scenarios must select either interval Hub power or minute backed load")
        if hub_intervals is None and realized.backed_load_kw is None:
            raise ValueError("EV scenarios require an issued Home Hub or minute backed-load forecast")
        horizon = max(horizon, self.forecast_length)
        step = realized.data.index[1] - realized.data.index[0]
        cfg = self.settings
        grid = pd.Timedelta(minutes=cfg["scenario_minutes"])
        if grid % step or timestamp != _round_instant(timestamp, grid, "floor"):
            raise ValueError("EV scenario origins/grid must align with the plant and scenario grid")
        mobility = realized.ev_schedules[self.vehicle_id]
        observed_arrival = last_observed_arrival(mobility, timestamp)
        lookback = pd.Timedelta(days=cfg["lookback_days"])
        history = pd.Timedelta(hours=cfg["knn_history_hours"])
        method = cfg["scenario_method"]
        left = timestamp-lookback
        # Require the full pool, not a silently shortened startup history.
        needed = left-history if method == "knn_state" else left
        if mobility.start > needed:
            raise ValueError("EV scenario replay lacks full lookback/history coverage; choose a later source start")
        current_home = bool(mobility.available([timestamp])[0])
        origins = pd.date_range(_round_instant(left, grid, "ceil"),
                                _round_instant(timestamp-horizon, grid, "floor"), freq=grid)
        ranked, excluded = [], 0
        # Vectorize history matching: a month of 15-minute candidates should
        # not require thousands of repeated pandas/trip traversals per solve.
        known_index = pd.date_range(_round_instant(needed, grid, "ceil"), timestamp, freq=grid)
        known_state = mobility.available(known_index)
        positions = known_index.get_indexer(origins)
        valid = np.ones(len(origins), dtype=bool)
        for trip in mobility.trips.loc[(mobility.trips.departure < timestamp) &
                                       (mobility.trips.arrival > timestamp)].itertuples():
            valid &= ~(origins+horizon > trip.departure)
        excluded = int((~valid).sum())
        if method == "knn_state":
            offsets = np.arange(1, int(history/grid)+1)
            observed = known_state[len(known_index)-1-offsets]
            distances = np.mean(known_state[positions[:, None]-offsets] != observed, axis=1)
        for i in range(len(origins)-1, -1, -1):
            origin = origins[i]
            # A bout crossing the path edge must also have a historically known
            # return/distance. Do not inspect its future distance as forecast data.
            if not valid[i]:
                continue
            home = bool(known_state[positions[i]])
            if method == "knn_state":
                distance = float(distances[i])
                tier = 0
            else:
                matches = dict(tod=origin.time() == timestamp.time(),
                    dow=origin.dayofweek == timestamp.dayofweek, state=home == current_home)
                tiers = [i for i, fields in enumerate(FILTERS[method]) if all(matches[f] for f in fields)]
                if not tiers:
                    continue
                tier, distance = min(tiers), 0.0
            ranked.append((tier, distance, origin))
        ranked.sort(key=lambda r: (r[0], r[1], -r[2].value))
        selected, overlap_rejected, used_days = [], 0, set()
        for tier, distance, origin in ranked:
            if method != "knn_state" and origin.date() in used_days:
                continue
            if any(origin < other+horizon and other < origin+horizon for _, _, other in selected):
                overlap_rejected += 1
                continue
            selected.append((tier, distance, origin))
            used_days.add(origin.date())
            if len(selected) == cfg["scenario_count"]:
                break
        load_source = (f"issued Home Hub interval forecast: {hub_intervals.source}"
                       if hub_intervals is not None else "shared minute backed-load forecast")
        diagnostics = dict(timestamp=str(timestamp), method=method, requested_count=cfg["scenario_count"],
            mobility_replay_metadata=mobility.metadata,
            scenario_count=len(selected), no_scenarios_selected=not selected, lookback_days=cfg["lookback_days"],
            forecast_hours=horizon.total_seconds()/3600, knn_history_hours=cfg["knn_history_hours"],
            candidate_count=len(ranked), incomplete_bout_exclusions=excluded,
            overlap_rejections=overlap_rejected, prefix_control_intervals=cfg["prefix_control_intervals"],
            probability_policy="uniform", backed_load_source=load_source)
        if hub_intervals is not None:
            diagnostics.update(hub_forecast_issued_at=str(hub_intervals.issued_at),
                               hub_forecast_inputs_ready_at=str(hub_intervals.inputs_ready_at))
        if not selected:
            self.issued_forecasts.append((timestamp, horizon, ()))
            self.history.append(dict(diagnostics, fallback_reason="no_complete_historical_scenarios"))
            raise NoEVScenarios("No complete non-overlapping historical EV scenarios")
        paths, forecasts, ids, records = [], [], [], []
        for tier, distance, origin in selected:
            source_home = bool(mobility.available([origin])[0])
            # The limited variant keeps matching historical paths intact,
            # including any home/away transitions inside the configured prefix.
            apply_persistence = (method != "tod_dow_state_hier_limit_persistence"
                                 or source_home != current_home)
            rows = mobility.trips.loc[(mobility.trips.arrival > origin) &
                                      (mobility.trips.departure < origin+horizon)].copy()
            for field in ("departure", "arrival"):
                rows[field] += timestamp-origin
            candidate_id = origin.isoformat()
            path = EVSchedule(rows, timestamp, timestamp+horizon+step,
                dict(candidate_id=candidate_id, source_origin=str(origin), source_end=str(origin+horizon),
                     filter_tier=tier, knn_distance=distance,
                     source_initial_state="home" if source_home else "away",
                     observed_initial_state="home" if current_home else "away",
                     last_observed_arrival=observed_arrival,
                     persistence_applied=apply_persistence))
            if apply_persistence:
                path = persistent_prefix(path, timestamp, horizon,
                    pd.Timedelta(minutes=self.control_minutes*cfg["prefix_control_intervals"]), current_home, step)
            paths.append(path)
            ids.append(candidate_id)
            records.append(path.metadata)
            forecasts.append(replace(realized, ev_schedules={**realized.ev_schedules, self.vehicle_id: path}, providers={}))
        diagnostics.update(candidate_ids=ids, candidates=records, optimization_probability=1/len(paths),
                           persistence_applied_count=sum(r["persistence_applied"] for r in records))
        self.history.append(diagnostics)
        self.issued_forecasts.append((timestamp, horizon, tuple(paths)))
        return EVScenarioSet(timestamp, tuple(paths), tuple(f.backed_load_kw for f in forecasts),
                             tuple(forecasts), tuple(ids), diagnostics)
