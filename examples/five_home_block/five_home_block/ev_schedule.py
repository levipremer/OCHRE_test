"""Small mobility adapters; distance is independent of the simulated battery.

Audi transition bins are conservatively unavailable. Contiguous away bins form
one home-to-home bout, including multiple episodes within a 15-minute bin.
Signed source SOC records supply energy for the configured power-loss replay;
distance and availability remain independent of that conversion.
"""
from dataclasses import dataclass, replace
from pathlib import Path
import datetime as dt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EVSchedule:
    """Canonical home-to-home trips (timestamps/km), replay bounds and provenance."""
    trips: pd.DataFrame
    start: pd.Timestamp
    end: pd.Timestamp
    metadata: dict

    def __post_init__(self):
        object.__setattr__(self, "start", pd.Timestamp(self.start))
        object.__setattr__(self, "end", pd.Timestamp(self.end))
        trips = self.trips.copy(deep=True)
        for column in ("departure", "arrival"):
            if column in trips:
                trips[column] = pd.to_datetime(trips[column])
        object.__setattr__(self, "trips", trips)
        required = {"departure", "arrival", "distance_km"}
        if not required <= set(self.trips) or self.end <= self.start:
            raise ValueError("EV mobility needs trip fields and positive coverage")
        if not self.trips.empty:
            if (not np.isfinite(self.trips.distance_km).all() or
                self.trips.distance_km.lt(0).any() or
                self.trips[["departure", "arrival"]].isna().any().any() or
                (self.trips.arrival <= self.trips.departure).any()):
                raise ValueError("EV trips require nonnegative finite distance and positive duration")
            if not self.trips.departure.is_monotonic_increasing or (
                self.trips.departure.iloc[1:].to_numpy() < self.trips.arrival.iloc[:-1].to_numpy()).any():
                raise ValueError("EV home-to-home trips must be ordered and nonoverlapping")
        object.__setattr__(self, "trips", normalize_ev_trips(trips))
        object.__setattr__(self, "metadata", {
            **self.metadata, "trip_time_rounding": "departure_floor_arrival_ceil",
            "trip_time_resolution": "1min",
        })

    def available(self, index):
        index = pd.DatetimeIndex(index)
        if len(index) and (index.min() < self.start or index.max() >= self.end):
            raise ValueError("EV availability requested outside validated coverage")
        result = np.ones(len(index), dtype=bool)
        for trip in self.trips.itertuples():
            result &= ~((index >= trip.departure) & (index < trip.arrival))
        return result

    def demand(self, consumption_kwh_per_km, *, soc_depletion_capacity_kwh=None):
        result = self.trips.copy(deep=True)
        if soc_depletion_capacity_kwh is None:
            result["energy_kwh"] = result.distance_km * consumption_kwh_per_km
        else:
            if "arrival_net_soc_depletion_pp" not in result and not result.empty:
                raise ValueError("Selected EV replay needs supplied arrival SOC depletion")
            values = (result.arrival_net_soc_depletion_pp.to_numpy(dtype=float)
                      if "arrival_net_soc_depletion_pp" in result else np.empty(0))
            if not np.isfinite(values).all() or not np.isfinite(soc_depletion_capacity_kwh) or soc_depletion_capacity_kwh <= 0:
                raise ValueError("Selected EV replay needs finite SOC depletion and capacity")
            result["energy_kwh"] = values * soc_depletion_capacity_kwh / 100
        return result


def normalize_ev_trips(rows, resolution="1min"):
    """Round away intervals outward and combine overlaps without changing demand.

    Distances and optional energy are summed exactly once when rounded intervals
    meet. Original timestamps are retained for audit and repeated normalization
    is idempotent. Physical and external-forecast schedules use one-minute bins.
    """
    if pd.Timedelta(resolution) != pd.Timedelta(minutes=1):
        raise ValueError("HEMS EV trip normalization requires one-minute resolution")
    if isinstance(rows, pd.DataFrame):
        columns = list(rows.columns)
        rows = rows.to_dict("records")
    else:
        rows = list(rows)
        columns = list(rows[0]) if rows else ["departure", "arrival", "distance_km"]
    if any("energy_kwh" in row for row in rows) and not all("energy_kwh" in row for row in rows):
        raise ValueError("Trip energy must be specified consistently for every trip")
    bouts = []
    for row in sorted(rows, key=lambda r: r["departure"]):
        row = dict(row)
        departure, arrival = pd.Timestamp(row["departure"]), pd.Timestamp(row["arrival"])
        if pd.isna(departure) or pd.isna(arrival) or arrival <= departure:
            raise ValueError("Trip duration must be positive")
        row["original_departures"] = tuple(row.get("original_departures", (departure.isoformat(),)))
        row["original_arrivals"] = tuple(row.get("original_arrivals", (arrival.isoformat(),)))
        merged_trips = row.get("merged_trips", 1)
        row["merged_trips"] = 1 if pd.isna(merged_trips) else int(merged_trips)
        row["departure"] = departure.floor(resolution)
        row["arrival"] = arrival.ceil(resolution)
        if not np.isfinite(row["distance_km"]) or row["distance_km"] < 0:
            raise ValueError("Trip distance must be finite and nonnegative")
        if "energy_kwh" in row and (not np.isfinite(row["energy_kwh"]) or row["energy_kwh"] < 0):
            raise ValueError("Trip energy must be finite and nonnegative")
        if bouts and row["departure"] <= bouts[-1]["arrival"]:
            bouts[-1]["arrival"] = max(bouts[-1]["arrival"], row["arrival"])
            bouts[-1]["distance_km"] += row["distance_km"]
            if "energy_kwh" in row:
                bouts[-1]["energy_kwh"] += row["energy_kwh"]
            for field in ("arrival_net_soc_depletion_pp", "arrival_inferred_soc_depletion_pp"):
                if field in row:
                    bouts[-1][field] += row[field]
            if "arrival_soc_depletion_inferred_count" in row:
                bouts[-1]["arrival_soc_depletion_inferred_count"] += row[
                    "arrival_soc_depletion_inferred_count"]
            bouts[-1]["merged_trips"] = bouts[-1].get("merged_trips", 1) + row.get("merged_trips", 1)
            bouts[-1]["original_departures"] += row["original_departures"]
            bouts[-1]["original_arrivals"] += row["original_arrivals"]
        else:
            bouts.append(row)
    result = pd.DataFrame(bouts, columns=None if bouts else columns)
    result.index.name = "trip_id"
    return result


def _bouts(rows, resolution):
    """Backward-compatible adapter entry point for the canonical minute policy."""
    return normalize_ev_trips(rows)


def delay_trip_returns(trips, minutes, *, start=None, end=None, collision_policy="error"):
    """Delay only returns, then select complete trips intersecting coverage.

    Apply to the source's complete trips before clipping to a replay/history
    window. Departures, signed SOC depletion, distance, and all demand/provenance
    fields are copied unchanged. Trips crossing a coverage edge remain complete.
    Reject collisions (including touching trips) before the canonical normalizer
    could merge bouts and lose the one-to-one comparison. The explicit approved
    cap policy keeps at least one home minute before the next departure. Zero is
    an exact copy. Only trips intersecting the requested coverage can be capped.

    This changes timing only; it does not freeze the existing power-loss model's
    conversion of source SOC percentage points into kWh at current capacity.
    """
    if type(minutes) is not int or minutes < 0:
        raise ValueError("EV return_delay_minutes must be a nonnegative whole minute integer")
    if collision_policy not in {"error", "cap_before_next_departure"}:
        raise ValueError("EV return_delay_collision_policy must be error or cap_before_next_departure")
    result = pd.DataFrame(trips).copy(deep=True)
    if result.empty and not len(result.columns):
        result = pd.DataFrame(columns=["departure", "arrival", "distance_km"])
    if minutes == 0 and start is None and end is None:
        return result
    if not result.empty:
        for column in ("departure", "arrival"):
            result[column] = pd.to_datetime(result[column])
    if minutes and not result.empty:
        if "undelayed_arrival" in result or "return_delay_minutes" in result:
            raise ValueError("EV return delay must be applied exactly once to source trips")
        result["undelayed_arrival"] = result.arrival.copy()
        result["return_delay_minutes"] = minutes
        result["actual_return_delay_minutes"] = float(minutes)
        result["return_delay_capped"] = False
        result["return_delay_next_departure"] = pd.NaT
        result["arrival"] += pd.Timedelta(minutes=minutes)
    selected = np.ones(len(result), dtype=bool)
    if start is not None and not result.empty:
        selected &= result.arrival.gt(pd.Timestamp(start)).to_numpy()
    if end is not None and not result.empty:
        selected &= result.departure.lt(pd.Timestamp(end)).to_numpy()
    if minutes and len(result) > 1:
        positions = np.argsort(result.departure.to_numpy(), kind="stable")
        ordered = result.iloc[positions]
        active = selected[positions]
        for i in range(len(ordered) - 1):
            current, following = ordered.iloc[i], ordered.iloc[i + 1]
            if (active[i] and
                    current.arrival.ceil("min") >= following.departure.floor("min")):
                if collision_policy == "cap_before_next_departure":
                    capped = following.departure.floor("min") - pd.Timedelta(minutes=1)
                    if capped < current.undelayed_arrival.ceil("min"):
                        raise ValueError("EV return-delay cap cannot preserve the original return "
                                         "and a full home minute before the next departure")
                    position = positions[i]
                    result.iloc[position, result.columns.get_loc("arrival")] = capped
                    result.iloc[position, result.columns.get_loc("actual_return_delay_minutes")] = (
                        capped - current.undelayed_arrival).total_seconds() / 60
                    result.iloc[position, result.columns.get_loc("return_delay_capped")] = True
                    result.iloc[position, result.columns.get_loc("return_delay_next_departure")] = following.departure
                    continue
                raise ValueError(
                    f"EV return delay {minutes} minutes causes a trip overlap: "
                    f"departure {current.departure}, original return {current.undelayed_arrival}, "
                    f"delayed return {current.arrival}, next departure {following.departure}. "
                    "Cannot preserve every departure and individual trip demand; "
                    "choose an explicit collision policy before running this variant.")
        if start is not None:
            selected &= result.arrival.gt(pd.Timestamp(start)).to_numpy()
    return result.loc[selected].copy(deep=True)


def _return_delay_metadata(trips, options):
    """Compact replay audit for requested delay and each approved exception."""
    frame = pd.DataFrame(trips)
    capped = (frame.loc[frame.return_delay_capped.eq(True)]
              if "return_delay_capped" in frame else frame.iloc[:0])
    records = [dict(departure=row.departure.isoformat(),
                    undelayed_arrival=row.undelayed_arrival.isoformat(),
                    arrival=row.arrival.isoformat(),
                    next_departure=row.return_delay_next_departure.isoformat(),
                    requested_delay_minutes=options.return_delay_minutes,
                    actual_delay_minutes=float(row.actual_return_delay_minutes))
               for row in capped.itertuples()]
    return dict(return_delay_minutes=options.return_delay_minutes,
                return_delay_scope="all_replay_history_and_future",
                return_delay_collision_policy=options.return_delay_collision_policy,
                return_delay_capped_trip_count=len(records), return_delay_caps=records)


def audi_schedule(path, ev, config):
    """Select a seeded, weekday/time-aligned interior window; never wrap Audi."""
    o = ev.hems
    path = Path(path)
    data = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    required = {"vehicle_key", "timestamp_local", "is_home", "arrival_distance_km", "arrival_indicator"}
    if not required <= set(data):
        raise ValueError(f"Audi input missing {sorted(required - set(data))}")
    data["timestamp_local"] = pd.to_datetime(data.timestamp_local, utc=True).dt.tz_convert("America/Los_Angeles")
    keys = [o.source_vehicle] if o.source_vehicle else np.random.default_rng(o.seed).permutation(sorted(data.vehicle_key.unique()))
    rejected = []
    for key in keys:
        try:
            result = _audi_vehicle_schedule(data, replace(ev, hems=replace(o, source_vehicle=str(key))), config)
            if o.soc_depletion_reference_capacity_kwh is not None:
                # Keep signed percentage records in canonical mobility; demand()
                # converts them after normalization, including negative energy.
                result = replace(result, metadata={**result.metadata,
                              "soc_depletion_reference_capacity_kwh": o.soc_depletion_reference_capacity_kwh,
                              "soc_depletion_basis": "original_signed_soc_percentage_points_fixed_reference_capacity"})
            return replace(result, metadata={**result.metadata, "source_file": str(path.resolve()),
                                             "excluded_vehicle_windows": rejected})
        except ValueError as error:
            if o.source_vehicle:
                raise
            rejected.append({"source_vehicle": str(key), "reason": str(error)})
    raise ValueError("No Audi vehicle has a complete interior window for the configured history, simulation and forecast")


def _audi_vehicle_schedule(data, ev, config):
    o = ev.hems
    rng = np.random.default_rng(o.seed)
    key = o.source_vehicle
    data = data.loc[data.vehicle_key.eq(key)].sort_values("timestamp_local").copy()
    if data.empty or data.timestamp_local.duplicated().any():
        raise ValueError("Audi vehicle missing or timestamps duplicated")
    if data.is_home.isna().any() or not data.is_home.isin([0, 1]).all():
        raise ValueError("Audi availability must be 0 or 1")
    if not data.timestamp_local.diff().dropna().eq(pd.Timedelta(minutes=15)).all():
        raise ValueError("Audi vehicle has gaps in its 15-minute timeline")
    # Preserve local clocks. Candidate windows crossing a DST change are excluded
    # explicitly instead of collapsing duplicate clocks or shifting trips by an hour.
    data["local"] = data.timestamp_local.dt.tz_localize(None)
    history = pd.Timedelta(days=o.history_days)
    future = pd.Timedelta(hours=max(o.horizon_hours, o.forecast_margin_hours))
    start = pd.Timestamp(config.start_time)
    duration = pd.Timedelta(config.duration)
    lower = data.local.min() + history
    upper = data.local.max() + pd.Timedelta(minutes=15) - duration - future
    candidates = data.loc[(data.local >= lower) & (data.local <= upper) & data.is_home.eq(1)
                          & data.local.dt.dayofweek.eq(start.dayofweek)
                          & data.local.dt.hour.eq(start.hour) & data.local.dt.minute.eq(start.minute)]
    candidates = candidates.iloc[len(candidates)//4:len(candidates) - len(candidates)//4]
    rows, source_soc_replacements = [], {}
    away_start = None
    for i, row in enumerate(data.itertuples()):
        if row.is_home == 0 and away_start is None:
            away_start = i
        if row.is_home == 1 and away_start is not None:
            if away_start > 0:  # incomplete leading bout is not invented
                # Some arrivals are recorded in the first fully home bin.
                bout = data.iloc[away_start:i+1]
                arrivals = bout.loc[bout.arrival_indicator.eq(1)]
                values = arrivals.arrival_distance_km
                distance = values.sum(min_count=1) if values.notna().all() else np.nan
                item = dict(departure=bout.local.iloc[0], arrival=row.local,
                            distance_km=float(distance), source="audi")
                for field in ("arrival_net_soc_depletion_pp", "arrival_inferred_soc_depletion_pp"):
                    if field in arrivals:
                        item[field] = arrivals[field].sum(min_count=1)
                if o.missing_soc_depletion_policy == "inferred":
                    # Resolve each source arrival before summing a home-to-home
                    # bout; summing observed values first can hide partial gaps.
                    # The raw source remains unchanged and the count travels
                    # with the effective depletion used by plant and MPC.
                    observed = (arrivals["arrival_net_soc_depletion_pp"].to_numpy(dtype=float)
                                if "arrival_net_soc_depletion_pp" in arrivals
                                else np.full(len(arrivals), np.nan))
                    inferred = (arrivals["arrival_inferred_soc_depletion_pp"].to_numpy(dtype=float)
                                if "arrival_inferred_soc_depletion_pp" in arrivals
                                else np.full(len(arrivals), np.nan))
                    missing = np.isnan(observed)
                    if not np.isfinite(observed[~missing]).all() or not np.isfinite(inferred[missing]).all():
                        raise ValueError("Audi arrival needs finite observed or inferred SOC depletion")
                    item["arrival_net_soc_depletion_pp"] = float(
                        np.where(missing, inferred, observed).sum())
                    item["arrival_soc_depletion_inferred_count"] = int(missing.sum())
                    replacements = []
                    for position in np.flatnonzero(missing):
                        source_arrival = arrivals.iloc[int(position)]
                        replacements.append({
                            "source_arrival_time": source_arrival.local.isoformat(),
                            "source_episode_id": (str(source_arrival.episode_id)
                                                  if "episode_id" in arrivals
                                                  and pd.notna(source_arrival.episode_id) else None),
                            "source_distance_km": float(source_arrival.arrival_distance_km),
                            "observed_net_soc_depletion_pp": None,
                            "supplied_inferred_soc_depletion_pp": float(inferred[position]),
                            "effective_net_soc_depletion_pp": float(inferred[position]),
                            "replacement_source_field": "arrival_inferred_soc_depletion_pp",
                            "provenance": "source distance-based SOC inference; no away charging assumed",
                        })
                    source_soc_replacements[(item["departure"], item["arrival"])] = replacements
                if "is_donor_reconstructed" in bout:
                    item["is_donor_reconstructed"] = int(bout.is_donor_reconstructed.max())
                if "arrival_possible_away_charging" in arrivals:
                    item["arrival_possible_away_charging"] = arrivals.arrival_possible_away_charging.max()
                if "episode_id" in arrivals:
                    item["source_episode_ids"] = "|".join(arrivals.episode_id.dropna().astype(str))
                rows.append(item)
            away_start = None
    for position in rng.permutation(len(candidates)):
        source_start = candidates.iloc[position].local
        left, right = source_start - history, source_start + duration + future
        window = data.loc[(data.local >= left) & (data.local < right)]
        if window.empty or not window.local.diff().dropna().eq(pd.Timedelta(minutes=15)).all():
            continue
        selected = [r.copy() for r in rows if r["arrival"] > left and r["departure"] < right]
        if any(not np.isfinite(r["distance_km"]) or r["distance_km"] < 0 for r in selected):
            continue
        # Trips touching a requested edge still need their complete demand/return.
        if away_start is not None and data.local.iloc[away_start] < right:
            continue
        if data.is_home.iloc[0] == 0 and left < data.loc[data.is_home.eq(1), "local"].iloc[0]:
            continue
        if o.return_delay_minutes:
            # Keep the baseline's chosen source window, and transform the full
            # source before selecting its history/future coverage. A return just
            # before the left edge can become an ongoing trip after the shift.
            required_left = left - pd.Timedelta(minutes=o.return_delay_minutes)
            if (required_left < data.local.min() or
                    (data.is_home.iloc[0] == 0 and required_left <
                     data.loc[data.is_home.eq(1), "local"].iloc[0])):
                raise ValueError("Audi return delay lacks complete source mobility before history coverage")
            selected = delay_trip_returns(rows, o.return_delay_minutes, start=left, end=right,
                                          collision_policy=o.return_delay_collision_policy).to_dict("records")
        # Audi candidates use naive local clocks. Shift on the destination's
        # local clock too, then attach its timezone so aware OCHRE simulations
        # retain the same weekday and time-of-day selection as naive runs.
        start_wall = start.tz_localize(None) if start.tz is not None else start
        offset = start_wall - source_start
        def mapped_time(source_time):
            wall = source_time + offset
            return (wall.tz_localize(start.tz, ambiguous="raise", nonexistent="raise")
                    if start.tz is not None else wall)
        replacements = []
        for r in selected:
            source_departure = r["departure"]
            source_arrival = r.get("undelayed_arrival", r["arrival"])
            for replacement in source_soc_replacements.get((source_departure, source_arrival), ()):
                replacements.append({
                    **replacement,
                    "simulation_source_arrival_time": mapped_time(
                        pd.Timestamp(replacement["source_arrival_time"])).isoformat(),
                    "source_bout_departure": source_departure.isoformat(),
                    "source_bout_arrival": source_arrival.isoformat(),
                    "simulation_bout_departure": mapped_time(source_departure).isoformat(),
                    "simulation_bout_arrival": mapped_time(source_arrival).isoformat(),
                })
            # Do not silently choose one repeated hour or create a missing
            # hour when the destination window crosses a clock change.
            r["departure"] = mapped_time(r["departure"])
            r["arrival"] = mapped_time(r["arrival"])
            if "undelayed_arrival" in r:
                r["undelayed_arrival"] = mapped_time(r["undelayed_arrival"])
            if pd.notna(r.get("return_delay_next_departure", pd.NaT)):
                r["return_delay_next_departure"] = mapped_time(r["return_delay_next_departure"])
        return EVSchedule(_bouts(selected, config.time_step), mapped_time(left),
                          mapped_time(right),
                          dict(source="audi", source_vehicle=key, source_start=str(source_start),
                               seed=o.seed, history_days=o.history_days,
                               forecast_margin_hours=future.total_seconds()/3600,
                               missing_soc_depletion_policy=o.missing_soc_depletion_policy,
                               **_return_delay_metadata(selected, o),
                               soc_depletion_replacements=replacements,
                               inferred_soc_arrival_count=sum(
                                   r.get("arrival_soc_depletion_inferred_count", 0) for r in selected),
                               inferred_soc_bout_count=sum(
                                   r.get("arrival_soc_depletion_inferred_count", 0) > 0 for r in selected)))
    raise ValueError("No complete Audi interior window covers history + simulation + forecast; "
                     "adjust windows or explicitly select another source_vehicle")


def nhts_schedule(path, ev, config):
    """Reuse existing NHTS cleaning/sampling, then wrap a fixed annual realization."""
    from .home_factory import build_trip_pool, build_ev_multiday_schedule, _load_weather_raw
    o = ev.hems
    # A fixed non-leap reference year makes the sampled mobility independent of
    # simulation length, battery capacity, and the number of MPC solves.
    origin = pd.Timestamp("2018-01-01")
    annual = replace(config, start_time=origin.to_pydatetime(), duration=dt.timedelta(days=367))
    pool = build_trip_pool(Path(path), config.NHTS_location_CDIVMSAR_ID)
    _, log = build_ev_multiday_schedule(ev, annual, pool, _load_weather_raw(config.weather_file),
                                       np.random.default_rng(o.seed))
    rows = []
    if not log.empty:
        for r in log.loc[log.trip_start_time.notna()].itertuples():
            departure = pd.Timestamp(r.trip_start_time)
            if origin <= departure < origin + pd.Timedelta(days=365):
                rows.append(dict(departure=departure, arrival=pd.Timestamp(r.trip_end_time),
                                 distance_km=float(r.trip_distance_miles)*1.609344, source="nhts"))
    annual_trips = _bouts(rows, config.time_step)
    start = pd.Timestamp(config.start_time)
    left = start - pd.Timedelta(days=o.history_days)
    right = start + config.duration + pd.Timedelta(hours=max(o.horizon_hours, o.forecast_margin_hours))
    anchor = pd.Timestamp(year=start.year, month=1, day=1)
    selected = []
    first = int(np.floor((left - pd.Timedelta(minutes=o.return_delay_minutes) - anchor)/pd.Timedelta(days=365))) - 1
    last = int(np.ceil((right - anchor)/pd.Timedelta(days=365))) + 1
    for cycle in range(first, last + 1):
        shift = anchor - origin + pd.Timedelta(days=365*cycle)
        for r in annual_trips.to_dict("records"):
            departure, arrival = r["departure"] + shift, r["arrival"] + shift
            selected.append({**r, "departure": departure, "arrival": arrival})
    selected = delay_trip_returns(selected, o.return_delay_minutes, start=left, end=right,
                                  collision_policy=o.return_delay_collision_policy)
    return EVSchedule(_bouts(selected, config.time_step), left, right,
                      dict(source="nhts", seed=o.seed, annual_period_days=365,
                           **_return_delay_metadata(selected, o)))


def build_ev_schedules(config, home):
    """Prepare one immutable realized mobility schedule per opted-in EV."""
    from .ev_config import validate_home_evs
    validate_home_evs(home.evs)
    schedules = {}
    largest_horizon = max((ev.hems.horizon_hours for ev in home.evs if ev.hems), default=0)
    for ev in home.evs:
        if ev.hems is None:
            continue
        ev = replace(ev, hems=replace(ev.hems, forecast_margin_hours=max(
            ev.hems.forecast_margin_hours, largest_horizon)))
        path = Path(ev.hems.source_file) if ev.hems.source_file else config.ev_trip_file
        if not path.is_absolute():
            path = config.repo_root / path
        adapter = audi_schedule if ev.hems.source == "audi" else nhts_schedule
        schedules[ev.vehicle_id] = adapter(path, ev, config)
    return schedules
