"""Causal historical EV analogs with a state-persistence forecast prefix.

Selection mirrors scenario-based-EV-prediction-analysis, with completed-path
eligibility and non-overlap added. Mobility stays distance-based, not SOC-based.
Prices and backed loads are shared today; paired Forecast objects allow future
joint-HEMS planned loads without changing the scenario optimizer interface.
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
    "tod_dow_hier": (("tod", "dow"), ("tod",)),
    "tod_state_hier": (("tod", "state"), ("tod",)),
}
SCENARIO_DEFAULTS = dict(scenario_method="knn_state", scenario_count=5,
    lookback_days=30, knn_history_hours=24, scenario_minutes=15, prefix_control_intervals=4)


class NoEVScenarios(ValueError):
    """No valid historical forecast: controller must use its baseline fallback."""


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


def persistent_prefix(mobility, timestamp, horizon, duration, available, plant_step):
    """Correct mismatched current state on [now, boundary); debit departures.

    Matching paths are unchanged, including transitions within the prefix.
    Home: discard trips ending within the prefix; shorten a crossing trip to
    depart at its endpoint, retaining its full distance. Never compound trips.
    Away: extend the ongoing absence, with zero new energy for that already
    debited trip; suppress any new departures absorbed into that absence.
    On a mismatch, prefix=0 still anchors the first plant interval.
    """
    scenario_home = bool(mobility.available([timestamp])[0])
    mismatch = scenario_home != bool(available)
    metadata = dict(mobility.metadata, prefix_applied=mismatch,
        prefix_original_state="home" if scenario_home else "away",
        prefix_state="home" if available else "away", trip_energy_timing="departure")
    if not mismatch:
        return replace(mobility, metadata=dict(metadata, prefix_end=str(timestamp),
            original_distance_km=float(mobility.trips.distance_km.sum()),
            retained_distance_km=float(mobility.trips.distance_km.sum()),
            ignored_distance_km=0.0, ignored_trip_count=0,
            shifted_trip_count=0, shifted_trips=[], shifted_distance_outside_horizon_km=0.0))
    end = timestamp + horizon
    boundary = timestamp + max(duration, plant_step)
    trips = mobility.trips.loc[(mobility.trips.arrival > timestamp) &
                               (mobility.trips.departure < end)].copy()
    overwritten = trips.loc[trips.arrival <= boundary]
    ignored = float(overwritten.distance_km.sum())
    ignored_count = len(overwritten)
    retained = trips.loc[trips.arrival > boundary].copy()
    retained = retained.reset_index(drop=True)
    shifted = []
    if available and not retained.empty and retained.loc[0, "departure"] < boundary:
        row = retained.iloc[0]
        shifted.append(dict(original_departure=str(row.departure), departure=str(boundary),
                            arrival=str(row.arrival), distance_km=float(row.distance_km),
                            remaining_absence_fraction=float((row.arrival-boundary)/(row.arrival-row.departure))))
        retained.loc[0, "departure"] = boundary
    elif not available:
        # No fabricated home interval or second debit at the prefix endpoint.
        # Absorb crossing/touching absences until a genuine home gap is possible.
        return_time = boundary
        count = 0
        for row in retained.itertuples():
            if row.departure > return_time:
                break
            return_time = max(return_time, row.arrival)
            ignored += float(row.distance_km)
            ignored_count += 1
            count += 1
        retained = retained.iloc[count:]
        ongoing = pd.DataFrame([dict(departure=timestamp-plant_step,
                                    arrival=return_time, distance_km=0.0)])
        retained = pd.concat([ongoing, retained], ignore_index=True)
    remaining = float(retained.distance_km.sum())
    if not np.isclose(float(trips.distance_km.sum()), ignored+remaining):
        raise ValueError("EV prefix trip-distance accounting does not reconcile")
    metadata = dict(metadata, prefix_end=str(boundary),
        original_distance_km=float(trips.distance_km.sum()), ignored_distance_km=ignored,
        ignored_trip_count=ignored_count, retained_distance_km=remaining,
        shifted_trip_count=len(shifted), shifted_trips=shifted,
        shifted_distance_outside_horizon_km=float(retained.loc[retained.departure >= end, "distance_km"].sum()))
    return EVSchedule(retained, timestamp, end+plant_step, metadata)


class ScenarioEVForecast:
    """Per-EV forecast issuance; no future realized mobility may enter selection."""
    def __init__(self, vehicle_id, control):
        self.vehicle_id = vehicle_id
        self.control_minutes = control["control_minutes"]
        self.forecast_length = pd.Timedelta(hours=control.get("forecast_hours", control["horizon_hours"]))
        self.settings = scenario_settings(control["forecast"], self.control_minutes, control["horizon_hours"])
        self.history = []
        self.backed_load_source = "shared non_v2h_twin trajectory"
        # Store final forecasts, not future truth. Scoring happens after simulation.
        self.issued_forecasts = []

    def at(self, timestamp, realized, horizon):
        timestamp = pd.Timestamp(timestamp)
        horizon = max(horizon, self.forecast_length)
        step = realized.data.index[1] - realized.data.index[0]
        cfg = self.settings
        grid = pd.Timedelta(minutes=cfg["scenario_minutes"])
        if grid % step or timestamp != timestamp.floor(grid):
            raise ValueError("EV scenario origins/grid must align with the plant and scenario grid")
        mobility = realized.ev_schedules[self.vehicle_id]
        lookback = pd.Timedelta(days=cfg["lookback_days"])
        history = pd.Timedelta(hours=cfg["knn_history_hours"])
        method = cfg["scenario_method"]
        left = timestamp-lookback
        # Require the full pool, not a silently shortened startup history.
        needed = left-history if method == "knn_state" else left
        if mobility.start > needed:
            raise ValueError("EV scenario replay lacks full lookback/history coverage; choose a later source start")
        current_home = bool(mobility.available([timestamp])[0])
        origins = pd.date_range(left.ceil(grid), (timestamp-horizon).floor(grid), freq=grid)
        ranked, excluded = [], 0
        # Vectorize history matching: a month of 15-minute candidates should
        # not require thousands of repeated pandas/trip traversals per solve.
        known_index = pd.date_range(needed.ceil(grid), timestamp, freq=grid)
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
        diagnostics = dict(timestamp=str(timestamp), method=method, requested_count=cfg["scenario_count"],
            mobility_replay_metadata=mobility.metadata,
            scenario_count=len(selected), no_scenarios_selected=not selected, lookback_days=cfg["lookback_days"],
            forecast_hours=horizon.total_seconds()/3600, knn_history_hours=cfg["knn_history_hours"],
            candidate_count=len(ranked), incomplete_bout_exclusions=excluded,
            overlap_rejections=overlap_rejected, prefix_control_intervals=cfg["prefix_control_intervals"],
            probability_policy="uniform", backed_load_source=self.backed_load_source)
        if not selected:
            self.issued_forecasts.append((timestamp, horizon, ()))
            self.history.append(dict(diagnostics, fallback_reason="no_complete_historical_scenarios"))
            raise NoEVScenarios("No complete non-overlapping historical EV scenarios")
        paths, forecasts, ids, records = [], [], [], []
        for tier, distance, origin in selected:
            rows = mobility.trips.loc[(mobility.trips.arrival > origin) &
                                      (mobility.trips.departure < origin+horizon)].copy()
            for field in ("departure", "arrival"):
                rows[field] += timestamp-origin
            candidate_id = origin.isoformat()
            path = EVSchedule(rows, timestamp, timestamp+horizon+step,
                dict(candidate_id=candidate_id, source_origin=str(origin), source_end=str(origin+horizon),
                     filter_tier=tier, knn_distance=distance))
            path = persistent_prefix(path, timestamp, horizon,
                pd.Timedelta(minutes=self.control_minutes*cfg["prefix_control_intervals"]), current_home, step)
            paths.append(path)
            ids.append(candidate_id)
            records.append(path.metadata)
            forecasts.append(replace(realized, ev_schedules={**realized.ev_schedules, self.vehicle_id: path}, providers={}))
        diagnostics.update(candidate_ids=ids, candidates=records, optimization_probability=1/len(paths),
                           prefix_applied_count=sum(r["prefix_applied"] for r in records))
        self.history.append(diagnostics)
        self.issued_forecasts.append((timestamp, horizon, tuple(paths)))
        return EVScenarioSet(timestamp, tuple(paths), tuple(f.backed_load_kw for f in forecasts),
                             tuple(forecasts), tuple(ids), diagnostics)
