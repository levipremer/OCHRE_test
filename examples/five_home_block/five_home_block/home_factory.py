"""Create and validate configured OCHRE dwellings."""

from __future__ import annotations

import numpy as np

import datetime as dt
from pathlib import Path

import pandas as pd

from ochre import Dwelling
from ochre.Equipment import Equipment

from .config import HomeConfig, NeighborhoodConfig, EVConfig

class HomeValidationError(RuntimeError):
    """Raised when a dwelling does not contain the required all-electric loads."""


def _event_timestamp(day: dt.datetime, clock: dt.time) -> dt.datetime:
    return dt.datetime.combine(day.date(), clock, tzinfo=day.tzinfo)

def _load_weather_raw(weather_file: Path) -> pd.DataFrame:
    """Load EPW weather and build a (month, day, hour, minute) lookup table."""

    # EPW has 8 header lines; no column names
    df = pd.read_csv(weather_file, skiprows=8, header=None)

    # Extract required columns by EPW position
    weather = pd.DataFrame({
        "dry_bulb": df[6],
        "rel_humidity": df[8] / 100.0,
        "ghi": df[13],
        "dni": df[14],
        "dhi": df[15],
        "wind_speed": df[21],
        "month": df[1],
        "day": df[2],
        "hour": df[3] - 1,
        "minute": df[4],
    })

    # Build lookup table keyed by (month, day, hour, minute)
    return weather.set_index(["month", "day", "hour", "minute"])



def load_and_filter_nhts(path: Path, cdivmsar_target: int | None) -> pd.DataFrame:

    # Load trip data from 2022 NHTS
    df = pd.read_csv(path)

    # Filter to only include trips made by drivers of private vehicles
    df = df[df["TRIPMODE"] == 1]

    # Filter by vehicle type (1–4) corresponding to sedan, suv, van, and truck
    df = df[df["VEHTYPE"].isin([1, 2, 3, 4])]

    # Filter by CDIVMSAR, a measure of region in US and proximity to large cities and rail transportation, only if provided
    if cdivmsar_target is not None:
        df = df[df["CDIVMSAR"] == cdivmsar_target]

    # Normalize times
    df["STRTTIME"] = df["STRTTIME"].astype(str).str.zfill(4)
    df["ENDTIME"] = df["ENDTIME"].astype(str).str.zfill(4)

    df["start_time"] = df["STRTTIME"].apply(lambda x: dt.time(int(x[:2]), int(x[2:])))
    df["end_time"]   = df["ENDTIME"].apply(lambda x: dt.time(int(x[:2]), int(x[2:])))

    # (Sunday=1)
    df["day_of_week"] = df["TRAVDAY"]

    return df

def group_by_vehicle_day(df: pd.DataFrame):
    return df.groupby(["VEHCASEID"])


def chunk_round_trips(trips: pd.DataFrame) -> list[dict]:
    """
    Build trip chunks for each vehicle (and person if present).
    """
    # work on a copy to avoid mutating caller's DataFrame
    trips = trips.copy()

    # normalize keys to avoid float/scientific-notation identity issues
    trips["VEHCASEID"] = trips["VEHCASEID"].astype(str)
    if "PERSONID" in trips.columns:
        trips["PERSONID"] = trips["PERSONID"].astype(str)
        sort_keys = ["VEHCASEID", "PERSONID", "SEQ_TRIPID"]
    else:
        sort_keys = ["VEHCASEID", "SEQ_TRIPID"]

    # ensure SEQ_TRIPID is integer for correct ordering
    trips["SEQ_TRIPID"] = pd.to_numeric(trips.get("SEQ_TRIPID", 0), errors="coerce").fillna(0).astype(int)

    # stable global sort: vehicle -> person (if present) -> sequence
    trips = trips.sort_values(sort_keys, ascending=[True] * len(sort_keys), kind="mergesort").reset_index(drop=True)

    chunks = []
    current_chunk = []

    for _, row in trips.iterrows():
        origin_home = (row["WHYFROM"] == 1)
        dest_home   = (row["WHYTO"] == 1)

        # Start a new chunk when origin is home
        if origin_home:
            # If a chunk is already open, close it first (only if last dest was home)
            if current_chunk:
                if current_chunk[-1]["WHYTO"] == 1:
                    chunks.append(current_chunk)
                current_chunk = []
            current_chunk = [row]
            continue

        # If chunk is open, append row and close when destination is home
        if current_chunk:
            current_chunk.append(row)
            if dest_home:
                chunks.append(current_chunk)
                current_chunk = []
        else:
            # Row not part of a home to home sequence
            continue

    # If last chunk never closed at home, discard it (preserve original behavior)
    return chunks


def summarize_chunk(chunk: list[dict]) -> dict:
    df = pd.DataFrame(chunk)

    start_time = df["start_time"].iloc[0]
    end_time   = df["end_time"].iloc[-1]
    miles      = df["TRPMILES"].replace(-1, 0).sum()

    # Commute classification
    dests = df["WHYTO"].tolist()
    has_work_or_school = (3 in dests) or (6 in dests)

    trip_type = "commute" if has_work_or_school else "noncommute"

    travday = df["day_of_week"].iloc[0]
    is_weekend = travday in (1, 7)

    return {
        "vehicle_case_id": df["VEHCASEID"].iloc[0],
        "day_of_week": travday,
        "is_weekend": is_weekend,
        "trip_type": trip_type,
        "trip_start_time": start_time,
        "trip_end_time": end_time,
        "trip_distance_miles": miles,
    }

def build_trip_pool(path: Path, cdivmsar_target: int):
    df = load_and_filter_nhts(path, cdivmsar_target)
    groups = group_by_vehicle_day(df)
    trip_rows = []
    for vehicle_id, trips in groups:
        chunks = chunk_round_trips(trips)
        for chunk in chunks:
            trip_rows.append(summarize_chunk(chunk))
    return pd.DataFrame(trip_rows)

def determine_commute(ev, trip_pool, is_weekend, rng, commute_cache):
    # If cached, return base commute (no noise)
    if ev.vehicle_id in commute_cache:
        return commute_cache[ev.vehicle_id]

    # Filter commute candidates
    if is_weekend:
        commute_candidates = trip_pool[
            (trip_pool["trip_type"] == "commute") &
            (trip_pool["is_weekend"] == True)
        ]
    else:
        commute_candidates = trip_pool[
            (trip_pool["trip_type"] == "commute") &
            (trip_pool["is_weekend"] == False)
        ]

    # Check for EVConfig-defined commute
    has_defined_times = (
        ev.nom_commute_arrival_time is not None and
        ev.nom_commute_departure_time is not None
    )
    has_defined_distance = ev.nom_commute_miles is not None

    # No commute available, then WFH
    if commute_candidates.empty:
        commute_cache[ev.vehicle_id] = {"work_from_home": True}
        return commute_cache[ev.vehicle_id]

    # Select commute from NHTS
    commute = commute_candidates.sample(1, random_state=rng).iloc[0]

    # Base commute distance
    if has_defined_distance:
        base_dist = ev.nom_commute_miles
        chosen_caseid = None
        base_start = commute["trip_start_time"]
        base_end   = commute["trip_end_time"]
    else:
        base_dist = commute["trip_distance_miles"]
        chosen_caseid = commute["vehicle_case_id"]
        base_start = commute["trip_start_time"]
        base_end   = commute["trip_end_time"]

    # Base commute times (no noise)
    if has_defined_times:
        base_arr = ev.nom_commute_arrival_time
        base_dep = ev.nom_commute_departure_time
    else:
        base_arr = base_end
        base_dep = base_start
        chosen_caseid = commute["vehicle_case_id"]

    # Store **base commute only** in cache
    commute_record = {
        "vehicle_case_id": chosen_caseid,
        "is_weekend": is_weekend,
        "trip_start_time": base_dep,
        "trip_end_time": base_arr,
        "trip_distance_miles": base_dist,
        "work_from_home": False,
    }

    commute_cache[ev.vehicle_id] = commute_record
    return commute_record


def determine_noncommute(
    ev: EVConfig,
    trip_pool: pd.DataFrame,
    is_weekend: bool,
    commute_info: dict,
    rng
) -> list[dict]:
    """
    Select all non-commute trip chunks for a randomly chosen vehicle_case_id.
    Filters by weekday/weekend before selecting the vehicle.
    """

    # Filter non-commute trips by weekday/weekend
    if is_weekend:
        candidates = trip_pool[
            (trip_pool["trip_type"] == "noncommute") &
            (trip_pool["is_weekend"] == True)
        ]
    else:
        candidates = trip_pool[
            (trip_pool["trip_type"] == "noncommute") &
            (trip_pool["is_weekend"] == False)
        ]

    # If none exist, return empty list
    if candidates.empty:
        return []

    # Select a random vehicle_case_id from filtered candidates
    vehicle_ids = candidates["vehicle_case_id"].unique()
    chosen_vehicle = rng.choice(vehicle_ids)

    # Select all non-commute chunks for that vehicle
    selected = candidates[candidates["vehicle_case_id"] == chosen_vehicle]

    # Convert rows to list of dicts, ignoring overlaps with commute
    trips = []
    for _, row in selected.iterrows():
        trip_dict = {
            "vehicle_case_id": row["vehicle_case_id"],
            "day_of_week": row["day_of_week"],
            "is_weekend": row["is_weekend"],
            "trip_start_time": row["trip_start_time"],
            "trip_end_time": row["trip_end_time"],
            "trip_distance_miles": row["trip_distance_miles"],
        }
        trips.append(trip_dict)

    return trips

def overlaps(trip, commute):
    trip_start = trip["trip_start_time"]
    trip_end   = trip["trip_end_time"]

    commute_start = commute["trip_start_time"]
    commute_end = commute["trip_end_time"]

    return not (trip_end <= commute_start or trip_start >= commute_end)

def distance_to_kwh(ev: EVConfig, distance_miles: float, temp_c: float) -> float:
    # Driving efficiencies of different EV vehicle types (miles/kWh)
    # https://www.epa.gov/system/files/documents/2026-02/420r26001.pdf
    base_eff = {
        "sedan": 3.4,
        "suv":   2.9,
        "truck": 2.2,
    }[ev.vehicle_type.lower()]

    # Temperature penalty (AAA test data)
    # https://newsroom.aaa.com/wp-content/uploads/2026/04/Research-Report_EV-vs-Hybrid-Efficiency_17-Apr-26_FINAL_HQ-2-1.pdf
    # 20°F (-6.7°C): 35.6% efficiency reduction
    # 75°F (23.9°C): baseline
    # 95°F (35.0°C): 10.4% efficiency reduction

    T_COLD = -6.7
    T_BASE = 23.9
    T_HOT  = 35.0

    LOSS_COLD = 0.356
    LOSS_BASE = 0.0
    LOSS_HOT  = 0.104

    # Linear interpolation between cold → base → hot
    if temp_c <= T_COLD:
        loss = LOSS_COLD
    elif temp_c <= T_BASE:
        loss = LOSS_COLD + (
            (LOSS_BASE - LOSS_COLD)
            * (temp_c - T_COLD)
            / (T_BASE - T_COLD)
        )
    elif temp_c <= T_HOT:
        loss = LOSS_BASE + (
            (LOSS_HOT - LOSS_BASE)
            * (temp_c - T_BASE)
            / (T_HOT - T_BASE)
        )
    else:
        loss = LOSS_HOT

    # Apply efficiency reduction
    eff = base_eff * (1.0 - loss)

    return distance_miles / eff

def build_ev_multiday_schedule(ev, neighborhood, trip_pool, weather_by_mdhm, rng):
    commute_cache = {}
    trip_log = []
    all_trips = []

    start = neighborhood.start_time
    end = start + neighborhood.duration
    days = pd.date_range(start=start, end=end, freq="D", tz=start.tzinfo)

    capacity_kwh = ev.capacity_kwh
    initial_soc = getattr(ev, "initial_soc_fraction", 0.5)
    wfh_prob = getattr(ev, "work_from_home_probability", 0.0)

    ev_static = {
        "ev_vehicle_id": ev.vehicle_id,
        "ev_capacity_kwh": ev.capacity_kwh,
        "ev_charger_power_kw": ev.charger_power_kw,
        "ev_initial_soc_fraction": initial_soc,
        "ev_work_from_home_probability": wfh_prob,
        "ev_vehicle_type": ev.vehicle_type,
        "ev_max_commute_distance_var_miles": ev.max_commute_distance_var_miles,
        "ev_max_commute_times_var_hr": ev.max_commute_times_var_hr,
        "ev_max_roundtrip_miles": ev.max_roundtrip_miles,
    }

    def avg_temp(day_dt, s, e):
        m, d = day_dt.month, day_dt.day
        h1, h2 = s.hour, e.hour
        return 0.5 * (
            weather_by_mdhm.loc[(m, d, h1, 0), "dry_bulb"] +
            weather_by_mdhm.loc[(m, d, h2, 0), "dry_bulb"]
        )

    def assign_trip_days(day_dt:dt.datetime, start_t: dt.datetime, end_t: dt.datetime) -> tuple[dt.datetime, dt.datetime]:
        start_day = day_dt
        end_day = day_dt + dt.timedelta(days=1) if end_t < start_t else day_dt
        return start_day, end_day


    def ts(day_dt, t):
        return _event_timestamp(day_dt, t)

    # ---------------------------------------------------------
    # PASS 1: Build all trips for all days
    # ---------------------------------------------------------

    #Filter out trips above a specified max trip distance
    if ev.max_roundtrip_miles is not None:
        trip_pool = trip_pool[trip_pool["trip_distance_miles"] <= ev.max_roundtrip_miles]

    for day in days:
        is_weekend = day.weekday() >= 5

        if rng.random() < wfh_prob:
            # Log WFH day
            trip_log.append({
                "day": day.date(),
                "trip_type": "WFH",
                "vehicle_case_id": None,
                "trip_start_time": None,
                "trip_end_time": None,
                "trip_distance_miles": 0.0,
                "avg_temp_C": None,
                "trip_kwh": 0.0,
                "soc_after_trip": None,
                "charge_start_time": None,
                "charge_end_time": None,
                **ev_static,
            })

            # Mark commute as skipped
            commute = {"work_from_home": True}
        else:
            commute = determine_commute(ev, trip_pool, is_weekend, rng, commute_cache)

        todays_trips = []

        if not commute.get("work_from_home", False):
            time_var = getattr(ev, "max_commute_times_var_hr", 0) or 0
            noisy_start = add_time_noise_earlier(commute["trip_start_time"], max_hours=time_var, rng=rng)
            noisy_end = add_time_noise_later(commute["trip_end_time"], max_hours=time_var, rng=rng)

            base_dist = commute["trip_distance_miles"]
            dist_var = getattr(ev, "max_commute_distance_var_miles", 0) or 0
            if dist_var <= 0:
                noisy_dist = base_dist
            else:
                noisy_dist = base_dist + rng.triangular(0, 0, dist_var)

            start_day, end_day = assign_trip_days(day, noisy_start, noisy_end)

            todays_trips.append({
                "start_day": start_day,
                "end_day": end_day,
                "trip_type": "commute",
                "vehicle_case_id": commute.get("vehicle_case_id"),
                "trip_start_time": noisy_start,
                "trip_end_time": noisy_end,
                "trip_distance_miles": noisy_dist,
            })

        noncommute = determine_noncommute(ev, trip_pool, is_weekend, commute, rng)
        for trip in noncommute:
            if not commute.get("work_from_home", False):
                if overlaps(trip, {"trip_start_time": noisy_start, "trip_end_time": noisy_end}):
                    continue

            start_day, end_day = assign_trip_days(day, trip["trip_start_time"], trip["trip_end_time"])

            todays_trips.append({
                "start_day": start_day,
                "end_day": end_day,
                "trip_type": "noncommute",
                "vehicle_case_id": trip["vehicle_case_id"],
                "trip_start_time": trip["trip_start_time"],
                "trip_end_time": trip["trip_end_time"],
                "trip_distance_miles": trip["trip_distance_miles"],
            })

        todays_trips.sort(key=lambda t: ts(t["start_day"], t["trip_start_time"]))
        all_trips.extend(todays_trips)

    # ---------------------------------------------------------
    # PASS 2: Build charging windows across all days
    # ---------------------------------------------------------
    events = []

    all_trips.sort(key=lambda tr: ts(tr["start_day"], tr["trip_start_time"]))

    if not all_trips:
        return pd.DataFrame([{
            "start_time": start,
            "end_time": end,
            "start_soc": initial_soc,
            "end_soc": 1.0,
        }]), pd.DataFrame(trip_log)

    first_trip = all_trips[0]
    first_start_ts = ts(first_trip["start_day"], first_trip["trip_start_time"])
    events.append({
        "start_time": ts(first_trip["start_day"], dt.time(0, 0)),
        "end_time": first_start_ts,
        "start_soc": initial_soc,
        "end_soc": 1.0,
    })

    for i, trip in enumerate(all_trips):
        start_day = trip["start_day"]
        end_day = trip["end_day"]
        start_ts = ts(start_day, trip["trip_start_time"])
        end_ts = ts(end_day, trip["trip_end_time"])

        tC = avg_temp(start_day, trip["trip_start_time"], trip["trip_end_time"])
        trip_kwh = distance_to_kwh(ev, trip["trip_distance_miles"], tC)
        soc_after_trip = max(1.0 - trip_kwh / capacity_kwh, 0.0)

        if i < len(all_trips) - 1:
            next_trip = all_trips[i + 1]
            next_ts = ts(next_trip["start_day"], next_trip["trip_start_time"])
        else:
            next_ts = end

        # ensure next_ts is strictly after end_ts
        while next_ts <= end_ts:
            next_ts = next_ts + dt.timedelta(days=1)

        orig_charge_start = end_ts
        orig_charge_end = next_ts

        clamped_start = max(orig_charge_start, start)
        clamped_end = min(orig_charge_end, end)

        if clamped_end <= clamped_start:
            continue

        events.append({
            "start_time": clamped_start,
            "end_time": clamped_end,
            "start_soc": soc_after_trip,
            "end_soc": 1.0,
        })

        trip_log.append({
            "day": start_day.date(),
            "trip_type": trip["trip_type"],
            "vehicle_case_id": trip["vehicle_case_id"],
            "trip_start_time": start_ts,
            "trip_end_time": end_ts,
            "trip_distance_miles": trip["trip_distance_miles"],
            "avg_temp_C": tC,
            "trip_kwh": trip_kwh,
            "soc_after_trip": soc_after_trip,
            "charge_start_time": clamped_start,
            "charge_end_time": clamped_end,
            "charge_start_time_original": orig_charge_start,
            "charge_end_time_original": orig_charge_end,
            "charge_clamped": (orig_charge_end != clamped_end or orig_charge_start != clamped_start),
            "clamp_reason": "horizon_clamp" if orig_charge_end > end else None,
            **ev_static,
        })

    return pd.DataFrame(events), pd.DataFrame(trip_log)




def add_time_noise_earlier(t: dt.time, max_hours: float, rng) -> dt.time:
    """Push departure earlier only using triangular distribution."""
    if max_hours <= 0:
        return t  # no noise

    delta = rng.triangular(-max_hours, 0, 0)# earlier only
    base = dt.datetime.combine(dt.date.today(), t)
    noisy = base + dt.timedelta(hours=delta)
    return noisy.time()


def add_time_noise_later(t: dt.time, max_hours: float, rng) -> dt.time:
    """Push arrival later only using triangular distribution."""
    if max_hours <= 0:
        return t  # no noise

    delta = rng.triangular(0, 0, max_hours)# later only
    base = dt.datetime.combine(dt.date.today(), t)
    noisy = base + dt.timedelta(hours=delta)
    return noisy.time()




def _equipment_overrides(
    home: HomeConfig,
    neighborhood: NeighborhoodConfig,
    rng,
    *,
    defrost_model: str = "Legacy",
    defrost_control_type: str = "Auto",
    defrost_er_strategy: str = "Aggressive",
) -> dict[str, dict]:

    heater = {
        "Deadband Temperature (C)": home.heating_deadband_c,
        "Deadband Offset (C)": home.heating_deadband_offset,
        "Compressor Minimum On Time (minutes)": home.compressor_min_on_time_min,
        "Compressor Minimum Off Time (minutes)": home.compressor_min_off_time_min,
        "Backup Deadband Temperature (C)": home.backup_deadband_c,
        "Backup Setpoint Offset (C)": home.backup_setpoint_offset_c,
        "Backup Stage Escalation Delay (minutes)": home.backup_stage_escalation_delay_min,
        "Defrost Model": defrost_model,
        "Defrost Control Type": defrost_control_type,
        "Defrost ER Strategy": defrost_er_strategy,
    }

    if home.backup_number_of_stages is not None:
        heater["Number of Backup Stages (-)"] = home.backup_number_of_stages

    if home.backup_stage_capacities_kw is not None:
        heater["Backup Stage Capacities (W)"] = [
            kw * 1000 for kw in home.backup_stage_capacities_kw
        ]

    # --- EV schedule integration

    trip_pool = build_trip_pool(
        neighborhood.ev_trip_file,
        neighborhood.NHTS_location_CDIVMSAR_ID,
    )

    weather_by_mdhm = _load_weather_raw(neighborhood.weather_file)

    from ochre.Equipment import ElectricVehicle

    ev_equipment = {}

    # Output directory for this home
    output_dir = neighborhood.output_directory / "data" / "raw" / home.home_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export log of trip_pool for this home
    trip_pool.to_csv(output_dir / "trip_pool.csv", index=False)

    for ev in home.evs:
        ev_sched, ev_log = build_ev_multiday_schedule(
            ev,
            neighborhood,
            trip_pool,
            weather_by_mdhm,
            rng,
        )

        # Export log of ev schedule
        (ev_log).to_csv(output_dir / f"EV_{ev.vehicle_id}_log.csv", index=False)

        ev_equipment[f"EV_{ev.vehicle_id}"] = {
            "equipment_class": ElectricVehicle,

            # Each EV must have its own end-use category
            "end_use": f"EV_{ev.vehicle_id}",

            # Required EV constructor args
            "vehicle_type": "BEV",
            "charging_level": "Level 2",
            "capacity": ev.capacity_kwh,
            "max_power": ev.charger_power_kw,

            # Your custom event schedule (must include start_time, end_time, start_soc, end_soc)
            "event_schedule": ev_sched,
        }

    return {
        "ASHP Heater": heater,
        "Water Heating": {"Setpoint Temperature (C)": home.water_heater_setpoint_c},
        **ev_equipment,
    }


def _set_thermostats(dwelling: Dwelling, home: HomeConfig) -> None:
    heater = dwelling.get_equipment_by_end_use("HVAC Heating")
    cooler = dwelling.get_equipment_by_end_use("HVAC Cooling")
    heater.schedule.loc[:, "HVAC Heating Setpoint (C)"] = home.heating_setpoint_c
    cooler.schedule.loc[:, "HVAC Cooling Setpoint (C)"] = home.cooling_setpoint_c
    heater.reset_time()
    cooler.reset_time()


def _single_equipment(dwelling: Dwelling, end_use: str, home_id: str) -> Equipment:
    equipment = dwelling.get_equipment_by_end_use(end_use)
    if equipment is None or isinstance(equipment, list):
        raise HomeValidationError(
            f"{home_id}: expected exactly one {end_use} device, got {equipment!r}"
        )
    return equipment


def validate_required_equipment(dwelling: Dwelling, home_id: str) -> None:
    """Ensure the configured dwelling is all-electric and has household loads."""
    required = ["HVAC Heating", "HVAC Cooling", "Water Heating"]
    for end_use in required:
        equipment = _single_equipment(dwelling, end_use, home_id)
        if not equipment.is_electric or equipment.is_gas:
            raise HomeValidationError(
                f"{home_id}: {end_use} device {equipment.name!r} is not all-electric"
            )

    excluded = set(required) | {"PV", "Battery", "Generator"}
    household_loads = [
        equipment
        for end_use, devices in dwelling.equipment_by_end_use.items()
        if end_use not in excluded
        for equipment in devices
        if equipment.is_electric
    ]
    if not household_loads:
        raise HomeValidationError(f"{home_id}: no non-HVAC household electric loads found")

    gas_devices = [equipment.name for equipment in dwelling.equipment.values() if equipment.is_gas]
    if gas_devices:
        raise HomeValidationError(f"{home_id}: fossil-fuel equipment found: {gas_devices}")


def create_dwelling(
    neighborhood: NeighborhoodConfig,
    home: HomeConfig,
    raw_output_directory: Path,
) -> Dwelling:
    """Instantiate one configured OCHRE dwelling using local repository inputs."""
    rng = np.random.default_rng(neighborhood.random_seed + home.seed_offset)
    raw_output_directory.mkdir(parents=True, exist_ok=True)
    dwelling = Dwelling(
        name=home.home_id,
        start_time=neighborhood.start_time,
        duration=neighborhood.duration,
        time_res=neighborhood.time_step,
        initialization_time=neighborhood.initialization_time,
        hpxml_file=str(home.hpxml_file),
        hpxml_schedule_file=str(home.schedule_file),
        weather_file=str(neighborhood.weather_file),
        output_path=str(raw_output_directory),
        save_results=True,
        save_args_to_json=True,
        verbosity=neighborhood.verbosity,
        metrics_verbosity=neighborhood.verbosity,
        seed=neighborhood.random_seed + home.seed_offset,
        Equipment=_equipment_overrides(
            home,
            neighborhood,
            rng,
            defrost_model=neighborhood.defrost_model,
            defrost_control_type=neighborhood.defrost_control_type,
            defrost_er_strategy=neighborhood.defrost_er_strategy,
        ),
    )
    _set_thermostats(dwelling, home)
    validate_required_equipment(dwelling, home.home_id)
    return dwelling
