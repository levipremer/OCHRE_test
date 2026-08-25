"""Create and validate configured OCHRE dwellings."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from ochre import Dwelling
from ochre.Equipment import Equipment

from .config import HomeConfig, NeighborhoodConfig


class HomeValidationError(RuntimeError):
    """Raised when a dwelling does not contain the required all-electric loads."""


def _event_timestamp(day: dt.datetime, clock: dt.time) -> dt.datetime:
    return dt.datetime.combine(day.date(), clock, tzinfo=day.tzinfo)


def build_ev_event(home: HomeConfig, start_time: dt.datetime) -> pd.DataFrame:
    """Build one explicit at-home EV charging event.

    OCHRE calls ``start_soc`` the SOC at arrival. If departure is earlier than
    arrival on the clock, it is interpreted as the following morning.
    """
    arrival = _event_timestamp(start_time, home.ev_arrival_time)
    departure = _event_timestamp(start_time, home.ev_departure_time)
    if departure <= arrival:
        departure += dt.timedelta(days=1)
    return pd.DataFrame(
        {
            "start_time": [arrival],
            "end_time": [departure],
            "start_soc": [home.ev_arrival_soc_fraction],
        }
    )


def _equipment_overrides(
    home: HomeConfig,
    start_time: dt.datetime,
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
        "Backup Stage Escalation Delay (minutes)": (
            home.backup_stage_escalation_delay_min
        ),
        "Defrost Model": defrost_model,
        "Defrost Control Type": defrost_control_type,
        "Defrost ER Strategy": defrost_er_strategy,
    }
    if home.backup_number_of_stages is not None:
        heater["Number of Backup Stages (-)"] = home.backup_number_of_stages
    if home.backup_stage_capacities_kw is not None:
        heater["Backup Stage Capacities (W)"] = [
            capacity * 1000 for capacity in home.backup_stage_capacities_kw
        ]

    return {
        # Equipment type and base sizing come from each home's ResStock HPXML.
        "ASHP Heater": heater,
        "Water Heating": {"Setpoint Temperature (C)": home.water_heater_setpoint_c},
        "Electric Vehicle": {
            "vehicle_type": "BEV",
            "charging_level": "Level 2",
            "capacity": home.ev_capacity_kwh,
            "max_power": home.ev_charger_power_kw,
            "event_schedule": build_ev_event(home, start_time),
        },
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
    required = ["HVAC Heating", "HVAC Cooling", "Water Heating", "EV"]
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
            neighborhood.start_time,
            defrost_model=neighborhood.defrost_model,
            defrost_control_type=neighborhood.defrost_control_type,
            defrost_er_strategy=neighborhood.defrost_er_strategy,
        ),
    )
    _set_thermostats(dwelling, home)
    validate_required_equipment(dwelling, home.home_id)
    return dwelling
