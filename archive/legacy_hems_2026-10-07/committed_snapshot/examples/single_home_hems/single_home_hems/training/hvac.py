"""Aligned HVAC and building-envelope training records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd


INDOOR_TEMPERATURE_COLUMN = "indoor_temperature_c"
NEXT_INDOOR_TEMPERATURE_COLUMN = "next_indoor_temperature_c"
OUTDOOR_TEMPERATURE_COLUMN = "outdoor_temperature_c"
GLOBAL_SOLAR_COLUMN = "global_horizontal_irradiance_w_per_m2"
DIRECT_NORMAL_IRRADIANCE_COLUMN = "direct_normal_irradiance_w_per_m2"
DIFFUSE_HORIZONTAL_IRRADIANCE_COLUMN = "diffuse_horizontal_irradiance_w_per_m2"
SOLAR_ZENITH_COLUMN = "solar_zenith_degrees"
SOLAR_AZIMUTH_COLUMN = "solar_azimuth_degrees"
DAY_INDICATOR_COLUMN = "internal_gain_day_indicator"
HVAC_HEAT_COLUMN = "hvac_indoor_sensible_heat_w"
HVAC_POWER_COLUMN = "hvac_electric_power_kw"


@dataclass(frozen=True)
class HVACInternalGainRegime:
    """Binary day/night regime used to infer two internal-gain levels."""

    day_start_hour: int = 6
    day_end_hour: int = 22

    def __post_init__(self) -> None:
        for name, hour in (
            ("day_start_hour", self.day_start_hour),
            ("day_end_hour", self.day_end_hour),
        ):
            if (
                isinstance(hour, bool)
                or not isinstance(hour, int)
                or not 0 <= hour <= 23
            ):
                raise ValueError(f"{name} must be an integer from 0 through 23")
        if self.day_start_hour == self.day_end_hour:
            raise ValueError("Day and night regimes require different boundary hours")

    def is_day(self, timestamp: pd.Timestamp) -> int:
        """Return one during the configured day regime and zero at night."""
        hour = pd.Timestamp(timestamp).hour
        if self.day_start_hour < self.day_end_hour:
            return int(self.day_start_hour <= hour < self.day_end_hour)
        return int(hour >= self.day_start_hour or hour < self.day_end_hour)


def _schedule_value(schedule: Mapping[str, Any], column: str) -> float:
    value = schedule.get(column)
    if value is None:
        raise ValueError(f"HVAC training schedule is missing {column!r}")
    return float(value)


def hvac_training_record(
    timestamp: pd.Timestamp,
    envelope: Any,
    heating: Any,
    cooling: Any,
    initial_indoor_temperature_c: float,
    gain_regime: HVACInternalGainRegime | None = None,
    source_schedule: Mapping[str, Any] | None = None,
) -> dict[str, float | str | int]:
    """Capture one building state transition and supporting diagnostics."""
    regime = gain_regime or HVACInternalGainRegime()
    schedule = envelope.current_schedule
    weather = source_schedule or schedule
    indoor = envelope.indoor_zone
    next_indoor_temperature = envelope.next_outputs[indoor.t_idx]
    occupancy = float(schedule.get("Occupancy (Persons)", 0))
    actual_internal_gain = (
        occupancy * envelope.occupancy_sensible_gain + indoor.internal_sens_gain
    )

    record: dict[str, float | str | int] = {
        INDOOR_TEMPERATURE_COLUMN: float(initial_indoor_temperature_c),
        OUTDOOR_TEMPERATURE_COLUMN: _schedule_value(
            schedule,
            "Ambient Dry Bulb (C)",
        ),
        GLOBAL_SOLAR_COLUMN: _schedule_value(weather, "GHI (W/m^2)"),
        DAY_INDICATOR_COLUMN: regime.is_day(timestamp),
        HVAC_HEAT_COLUMN: float(indoor.hvac_sens_gain),
        NEXT_INDOOR_TEMPERATURE_COLUMN: float(next_indoor_temperature),
        HVAC_POWER_COLUMN: float(heating.electric_kw + cooling.electric_kw),
        "actual_internal_sensible_gain_w": float(actual_internal_gain),
        "occupancy_persons": occupancy,
        "heating_setpoint_c": float(heating.temp_setpoint),
        "cooling_setpoint_c": float(cooling.temp_setpoint),
        "heating_mode": str(heating.mode),
        "cooling_mode": str(cooling.mode),
    }

    # Retain raw solar components and geometry for projected-DNI candidates.
    optional_diagnostics = {
        DIRECT_NORMAL_IRRADIANCE_COLUMN: "DNI (W/m^2)",
        DIFFUSE_HORIZONTAL_IRRADIANCE_COLUMN: "DHI (W/m^2)",
        SOLAR_ZENITH_COLUMN: "zenith",
        SOLAR_AZIMUTH_COLUMN: "azimuth",
        "ground_temperature_c": "Ground Temperature (C)",
        "wind_speed_m_per_s": "Wind Speed (m/s)",
        "ambient_humidity_ratio": "Ambient Humidity Ratio (-)",
        "sky_temperature_c": "Sky Temperature (C)",
    }
    for output_column, schedule_column in optional_diagnostics.items():
        value = weather.get(schedule_column)
        record[output_column] = float(value) if value is not None else float("nan")

    return record
