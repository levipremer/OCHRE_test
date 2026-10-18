"""Small mobility adapters; distance is independent of the simulated battery.

Audi transition bins are conservatively unavailable. Contiguous away bins form
one home-to-home bout, including multiple episodes within a 15-minute bin.
No signed SOC field is used to infer distance, energy, or availability.
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

    def available(self, index):
        index = pd.DatetimeIndex(index)
        if len(index) and (index.min() < self.start or index.max() >= self.end):
            raise ValueError("EV availability requested outside validated coverage")
        result = np.ones(len(index), dtype=bool)
        for trip in self.trips.itertuples():
            result &= ~((index >= trip.departure) & (index < trip.arrival))
        return result

    def demand(self, consumption_kwh_per_km):
        result = self.trips.copy(deep=True)
        result["energy_kwh"] = result.distance_km * consumption_kwh_per_km
        return result


def _bouts(rows, resolution):
    """Combine overlapping trips without losing distance; never invent charging gaps."""
    bouts = []
    for row in sorted(rows, key=lambda r: r["departure"]):
        row = dict(row)
        row["departure"] = pd.Timestamp(row["departure"]).floor(resolution)
        row["arrival"] = pd.Timestamp(row["arrival"]).ceil(resolution)
        if row["arrival"] <= row["departure"]:
            raise ValueError("Trip duration must be positive")
        if not np.isfinite(row["distance_km"]) or row["distance_km"] < 0:
            raise ValueError("Trip distance must be finite and nonnegative")
        if bouts and row["departure"] <= bouts[-1]["arrival"]:
            bouts[-1]["arrival"] = max(bouts[-1]["arrival"], row["arrival"])
            bouts[-1]["distance_km"] += row["distance_km"]
            bouts[-1]["merged_trips"] = bouts[-1].get("merged_trips", 1) + 1
        else:
            bouts.append(row)
    result = pd.DataFrame(bouts, columns=None if bouts else ["departure", "arrival", "distance_km"])
    result.index.name = "trip_id"
    return result


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
            return replace(result, metadata={**result.metadata, "excluded_vehicle_windows": rejected})
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
    rows = []
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
        offset = start - source_start
        for r in selected:
            r["departure"] += offset
            r["arrival"] += offset
        return EVSchedule(_bouts(selected, config.time_step), start - history,
                          start + duration + future,
                          dict(source="audi", source_vehicle=key, source_start=str(source_start),
                               seed=o.seed, history_days=o.history_days,
                               forecast_margin_hours=future.total_seconds()/3600))
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
    first = int(np.floor((left - anchor)/pd.Timedelta(days=365))) - 1
    last = int(np.ceil((right - anchor)/pd.Timedelta(days=365))) + 1
    for cycle in range(first, last + 1):
        shift = anchor - origin + pd.Timedelta(days=365*cycle)
        for r in annual_trips.to_dict("records"):
            departure, arrival = r["departure"] + shift, r["arrival"] + shift
            if arrival > left and departure < right:
                selected.append({**r, "departure": departure, "arrival": arrival})
    return EVSchedule(_bouts(selected, config.time_step), left, right,
                      dict(source="nhts", seed=o.seed, annual_period_days=365))


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
