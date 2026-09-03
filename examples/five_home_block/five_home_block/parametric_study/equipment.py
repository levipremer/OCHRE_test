"""Scenario-aware OCHRE dwelling construction without changing the baseline factory."""

from __future__ import annotations

import datetime as dt
import xml.etree.ElementTree as ET

import pandas as pd
import numpy as np

from ochre import Dwelling

from ..config import HomeConfig
from ..home_factory import _equipment_overrides, _set_thermostats, _single_equipment
from .config import ParametricStudyConfig, ResolvedSeason
from .cooking import FuelAwareCookingRange
from .scenarios import EquipmentState
from ..home_factory import (
    _equipment_overrides,
    _set_thermostats,
    _single_equipment,
    build_trip_pool,
    _load_weather_raw,
    build_ev_multiday_schedule,
)
from ..config import NeighborhoodConfig


BTU_PER_HOUR_TO_W = 0.29307107


def _hpxml_backup_heating_capacity_w(home: HomeConfig) -> float:
    """Read the existing ResStock backup/design capacity without changing HPXML."""
    root = ET.parse(home.hpxml_file).getroot()
    for tag_name in ("BackupHeatingCapacity", "HeatingCapacity"):
        values = [
            element.text
            for element in root.iter()
            if element.tag.rsplit("}", 1)[-1] == tag_name and element.text
        ]
        if values:
            return float(values[0]) * BTU_PER_HOUR_TO_W
    raise RuntimeError(f"{home.home_id}: HPXML has no heating capacity")


def build_daily_ev_events(
    home: HomeConfig, start_time: dt.datetime, end_time: dt.datetime
) -> pd.DataFrame:
    """Repeat existing per-home EV assumptions on every simulated day."""
    last_day = (end_time - dt.timedelta(microseconds=1)).date()
    days = pd.date_range(start_time.date(), last_day, freq="D")
    rows = []
    for day in days:
        arrival = dt.datetime.combine(day.date(), home.ev_arrival_time)
        departure = dt.datetime.combine(day.date(), home.ev_departure_time)
        if departure <= arrival:
            departure += dt.timedelta(days=1)
        rows.append(
            {
                "start_time": arrival,
                "end_time": departure,
                "start_soc": home.ev_arrival_soc_fraction,
            }
        )
    return pd.DataFrame(rows)

def build_scenario_equipment_overrides(
    study: ParametricStudyConfig,
    home: HomeConfig,
    period: ResolvedSeason,
    state: EquipmentState,
    rng,
) -> dict[str, dict]:
    """Layer one equipment state over the existing per-home controls."""
    base = study.base

    neighborhood = base.neighborhood

    # Build EV trip pool and weather for this scenario
    trip_pool = build_trip_pool(
        neighborhood.ev_trip_file,
        neighborhood.NHTS_location_CDIVMSAR_ID,
    )
    weather_by_mdhm = _load_weather_raw(neighborhood.weather_file)

    # Call the updated equipment overrides with all required arguments
    overrides = _equipment_overrides(
        home,
        period.run_start,
        defrost_model=base.defrost_model,
        defrost_control_type=base.defrost_control_type,
        defrost_er_strategy=base.defrost_er_strategy,
        rng=rng,
        neighborhood=neighborhood,
        trip_pool=trip_pool,
        weather_by_mdhm=weather_by_mdhm,
    )

    # --- Fuel switching for heating and water heating ---

    if state.heating == "gas":
        overrides.pop("ASHP Heater", None)
        overrides["HVAC Heating"] = {
            "Equipment Name": "Furnace",
            "Fuel": "Natural gas",
            "EIR (-)": 1 / study.gas_furnace.afue,
            "Rated Efficiency": f"{study.gas_furnace.afue:g} AFUE",
            "Number of Speeds (-)": 1,
            "Capacity (W)": _hpxml_backup_heating_capacity_w(home),
        }

    if state.water_heating == "gas":
        gas = study.gas_water_heater
        overrides["Water Heating"] = {
            "Equipment Name": "storage water heater",
            "Fuel": "Natural gas",
            "Setpoint Temperature (C)": home.water_heater_setpoint_c,
            "Energy Factor (-)": gas.energy_factor,
            "Efficiency (-)": gas.burner_efficiency,
            "Capacity (W)": gas.capacity_w,
            "UA (W/K)": gas.ua_w_per_k,
        }
    elif state.water_heating == "erwh":
        resistance = study.electric_resistance_water_heater
        overrides["Water Heating"] = {
            "Equipment Name": "storage water heater",
            "Fuel": "Electricity",
            "Setpoint Temperature (C)": home.water_heater_setpoint_c,
            "Efficiency (-)": resistance.efficiency,
            "Capacity (W)": resistance.capacity_w,
        }

    overrides.setdefault("Cooking Range", {})["equipment_class"] = FuelAwareCookingRange
    return overrides

def build_scenario_equipment_overrides(
    study: ParametricStudyConfig,
    home: HomeConfig,
    period: ResolvedSeason,
    state: EquipmentState,
    rng,
) -> dict[str, dict]:
    """Layer one equipment state over the existing per-home controls."""
    base = study.base

    # EV inputs come from base config, not from a neighborhood object
    trip_pool = build_trip_pool(
        base.ev_trip_file,
        base.NHTS_location_CDIVMSAR_ID,
    )

    overrides = _equipment_overrides(
        home,
        base,
        rng,
        defrost_model=base.defrost_model,
        defrost_control_type=base.defrost_control_type,
        defrost_er_strategy=base.defrost_er_strategy,
    )

    # --- Fuel switching overrides (unchanged) ---
    if state.heating == "gas":
        overrides.pop("ASHP Heater", None)
        overrides["HVAC Heating"] = {
            "Equipment Name": "Furnace",
            "Fuel": "Natural gas",
            "EIR (-)": 1 / study.gas_furnace.afue,
            "Rated Efficiency": f"{study.gas_furnace.afue:g} AFUE",
            "Number of Speeds (-)": 1,
            "Capacity (W)": _hpxml_backup_heating_capacity_w(home),
        }

    if state.water_heating == "gas":
        gas = study.gas_water_heater
        overrides["Water Heating"] = {
            "Equipment Name": "storage water heater",
            "Fuel": "Natural gas",
            "Setpoint Temperature (C)": home.water_heater_setpoint_c,
            "Energy Factor (-)": gas.energy_factor,
            "Efficiency (-)": gas.burner_efficiency,
            "Capacity (W)": gas.capacity_w,
            "UA (W/K)": gas.ua_w_per_k,
        }
    elif state.water_heating == "erwh":
        resistance = study.electric_resistance_water_heater
        overrides["Water Heating"] = {
            "Equipment Name": "storage water heater",
            "Fuel": "Electricity",
            "Setpoint Temperature (C)": home.water_heater_setpoint_c,
            "Efficiency (-)": resistance.efficiency,
            "Capacity (W)": resistance.capacity_w,
        }

    overrides.setdefault("Cooking Range", {})["equipment_class"] = FuelAwareCookingRange
    return overrides


def create_scenario_dwelling(
    study: ParametricStudyConfig,
    home: HomeConfig,
    period: ResolvedSeason,
    state: EquipmentState,
) -> Dwelling:
    """Instantiate a profile run while preserving base inputs and electric sizing."""
    base = study.base
    cooking_fuel = "natural gas" if state.cooking == "gas" else "electricity"
    dwelling = Dwelling(
        name=home.home_id,
        start_time=period.run_start,
        duration=period.duration,
        time_res=base.time_step,
        initialization_time=None,
        hpxml_file=str(home.hpxml_file),
        hpxml_schedule_file=str(home.schedule_file),
        weather_file=str(base.weather_file),
        output_path=None,
        save_results=False,
        save_args_to_json=False,
        verbosity=base.verbosity,
        metrics_verbosity=base.verbosity,
        seed=base.random_seed + home.seed_offset,
        modify_hpxml_dict={
            "Appliances": {"CookingRange": {"FuelType": cooking_fuel}}
        },
        Equipment=build_scenario_equipment_overrides(study, home, period, state, np.random.default_rng(base.random_seed + home.seed_offset)),
    )
    _set_thermostats(dwelling, home)
    validate_scenario_equipment(dwelling, home.home_id, state)
    return dwelling


def validate_scenario_equipment(
    dwelling: Dwelling, home_id: str, state: EquipmentState
) -> None:
    """Ensure instantiated equipment matches the requested fuels and technology."""
    heater = _single_equipment(dwelling, "HVAC Heating", home_id)
    water_heater = _single_equipment(dwelling, "Water Heating", home_id)
    cooking = dwelling.equipment.get("Cooking Range")
    if cooking is None:
        raise RuntimeError(f"{home_id}: Cooking Range equipment is missing")

    if state.heating == "gas":
        if heater.name != "Gas Furnace" or not heater.is_gas:
            raise RuntimeError(f"{home_id}: expected a gas furnace, got {heater.name}")
    elif heater.name != "ASHP Heater" or heater.is_gas:
        raise RuntimeError(f"{home_id}: expected configured ASHP heating, got {heater.name}")

    expected_water_names = {
        "gas": "Gas Water Heater",
        "hpwh": "Heat Pump Water Heater",
        "erwh": "Electric Resistance Water Heater",
    }
    if water_heater.name != expected_water_names[state.water_heating]:
        raise RuntimeError(
            f"{home_id}: expected {expected_water_names[state.water_heating]}, "
            f"got {water_heater.name}"
        )
    if state.water_heating == "gas" and not water_heater.is_gas:
        raise RuntimeError(f"{home_id}: gas water heater is not reporting gas capability")
    if state.water_heating != "gas" and water_heater.is_gas:
        raise RuntimeError(f"{home_id}: electric water heater reports gas capability")

    if state.cooking == "gas" and not cooking.is_gas:
        raise RuntimeError(f"{home_id}: gas cooking state is not gas capable")
    if state.cooking == "electric" and cooking.is_gas:
        raise RuntimeError(f"{home_id}: electric cooking state reports gas capability")
