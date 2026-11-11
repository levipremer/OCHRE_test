"""Aligned HVAC and building-envelope training records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


INDOOR_TEMPERATURE_COLUMN = "indoor_temperature_c"
NEXT_INDOOR_TEMPERATURE_COLUMN = "next_indoor_temperature_c"
OUTDOOR_TEMPERATURE_COLUMN = "outdoor_temperature_c"
OUTDOOR_RELATIVE_HUMIDITY_COLUMN = "outdoor_relative_humidity_fraction"
GLOBAL_SOLAR_COLUMN = "global_horizontal_irradiance_w_per_m2"
DIRECT_NORMAL_IRRADIANCE_COLUMN = "direct_normal_irradiance_w_per_m2"
DIFFUSE_HORIZONTAL_IRRADIANCE_COLUMN = "diffuse_horizontal_irradiance_w_per_m2"
SOLAR_ZENITH_COLUMN = "solar_zenith_degrees"
SOLAR_AZIMUTH_COLUMN = "solar_azimuth_degrees"
DAY_INDICATOR_COLUMN = "internal_gain_day_indicator"
NIGHT_INDICATOR_COLUMN = "internal_gain_night_indicator"
HVAC_HEAT_COLUMN = "hvac_indoor_sensible_heat_w"
HVAC_COIL_ER_HEAT_COLUMN = "hvac_coil_er_heat_w"
HVAC_POWER_COLUMN = "hvac_electric_power_kw"
HVAC_HEATING_POWER_COLUMN = "hvac_heating_electric_power_kw"
HVAC_PROXY_HEAT_COLUMN = "hvac_hp_fan_cop_proxy_w"
HVAC_HP_PROXY_COLUMN = "hvac_hp_fan_cop_proxy_hp_w"
HVAC_HP_MAIN_POWER_COLUMN = "hvac_hp_main_electric_power_w"
HVAC_FAN_POWER_COLUMN = "hvac_heating_fan_electric_power_w"
HVAC_HP_COIL_HEAT_COLUMN = "hvac_hp_coil_heat_w"
HVAC_ER_HEAT_COLUMN = "hvac_er_heat_w"

HEATING_SETPOINT_PERIODS = (
    {"start": "06:00", "end": "09:00", "setpoint_c": 22.5},
    {"start": "09:00", "end": "17:00", "setpoint_c": 18.0},
    {"start": "17:00", "end": "21:00", "setpoint_c": 22.5},
    {"start": "21:00", "end": "06:00", "setpoint_c": 20.5},
)


def heating_setpoints_c(index: pd.DatetimeIndex) -> np.ndarray:
    """Four-period thermostat profile used to train the HVAC predictor."""
    hours = pd.DatetimeIndex(index).hour.to_numpy()
    return np.select(
        ((hours >= 6) & (hours < 9), (hours >= 9) & (hours < 17),
         (hours >= 17) & (hours < 21)),
        (22.5, 18.0, 22.5),
        default=20.5,
    )


def heating_setpoint_c(timestamp: pd.Timestamp) -> float:
    """Return the training heating setpoint for one simulation timestep."""
    return float(heating_setpoints_c(pd.DatetimeIndex([pd.Timestamp(timestamp)]))[0])


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


def calculate_hvac_proxy_heat_w(
    *,
    hp_main_power_w: float,
    fan_power_w: float,
    hp_coil_heat_w: float,
    er_heat_w: float,
) -> tuple[float, float]:
    """Return HP and total model inputs from unscaled equipment components.

    The HP-specific coil COP is coil heat divided by HP main electricity.
    Multiplying that COP by HP main plus blower electricity matches the
    optimizer's HP heat proxy. Negative coil heat during defrost is retained.
    """
    main, fan, coil, er = map(
        float,
        (hp_main_power_w, fan_power_w, hp_coil_heat_w, er_heat_w),
    )
    if not np.isfinite((main, fan, coil, er)).all():
        raise ValueError("HVAC proxy components must be finite")
    if main < 0 or fan < 0 or er < 0:
        raise ValueError("HVAC proxy power and resistance heat cannot be negative")
    if main == 0 and coil > 0:
        raise ValueError("Positive HP coil heat requires positive HP main power")
    hp_proxy = (main + fan) * coil / main if main > 0 else 0.0
    total_proxy = hp_proxy + er
    if not np.isfinite((hp_proxy, total_proxy)).all():
        raise ValueError("HVAC proxy heat must be finite")
    return float(hp_proxy), float(total_proxy)


def _hvac_proxy_components_from_equipment(
    heating: Any,
) -> tuple[float, float, float, float]:
    """Read unscaled HP, fan, coil, and ER components after OCHRE's model step."""
    scale = float(heating.space_fraction)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("HVAC conditioned-space fraction must be positive and finite")
    er_heat_w = float(heating.er_capacity)
    fan_power_w = float(heating.fan_power) / scale
    hp_coil_heat_w = float(heating.capacity) - er_heat_w
    if str(heating.defrost_model) == "Discrete":
        hp_main_power_w = float(heating.hp_power)
    else:
        hp_main_power_w = (
            float(heating.capacity) * float(heating.eir)
            - er_heat_w * float(heating.er_eir_rated)
        )
        # Mixed-mode weighted EIR arithmetic can leave a sub-microwatt
        # residual when the compressor is truly off.
        if abs(hp_main_power_w) < 1e-6:
            hp_main_power_w = 0.0
    return hp_main_power_w, fan_power_w, hp_coil_heat_w, er_heat_w


def hvac_proxy_from_equipment(heating: Any) -> tuple[float, float]:
    """Read OCHRE's completed minute and return HP and total model inputs W."""
    main, fan, coil, er = _hvac_proxy_components_from_equipment(heating)
    return calculate_hvac_proxy_heat_w(
        hp_main_power_w=main,
        fan_power_w=fan,
        hp_coil_heat_w=coil,
        er_heat_w=er,
    )


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
    hp_proxy_w, total_proxy_w = hvac_proxy_from_equipment(heating)
    hp_main_power_w, fan_power_w, hp_coil_heat_w, er_heat_w = (
        _hvac_proxy_components_from_equipment(heating)
    )
    day_indicator = regime.is_day(timestamp)

    record: dict[str, float | str | int] = {
        INDOOR_TEMPERATURE_COLUMN: float(initial_indoor_temperature_c),
        OUTDOOR_TEMPERATURE_COLUMN: _schedule_value(
            schedule,
            "Ambient Dry Bulb (C)",
        ),
        OUTDOOR_RELATIVE_HUMIDITY_COLUMN: _schedule_value(
            weather,
            "Ambient Relative Humidity (-)",
        ),
        GLOBAL_SOLAR_COLUMN: _schedule_value(weather, "GHI (W/m^2)"),
        DAY_INDICATOR_COLUMN: day_indicator,
        NIGHT_INDICATOR_COLUMN: 1 - day_indicator,
        HVAC_HEAT_COLUMN: float(indoor.hvac_sens_gain),
        HVAC_COIL_ER_HEAT_COLUMN: hp_coil_heat_w + er_heat_w,
        HVAC_PROXY_HEAT_COLUMN: total_proxy_w,
        HVAC_HP_PROXY_COLUMN: hp_proxy_w,
        HVAC_HP_MAIN_POWER_COLUMN: hp_main_power_w,
        HVAC_FAN_POWER_COLUMN: fan_power_w,
        HVAC_HP_COIL_HEAT_COLUMN: hp_coil_heat_w,
        HVAC_ER_HEAT_COLUMN: er_heat_w,
        NEXT_INDOOR_TEMPERATURE_COLUMN: float(next_indoor_temperature),
        HVAC_POWER_COLUMN: float(heating.electric_kw + cooling.electric_kw),
        HVAC_HEATING_POWER_COLUMN: float(heating.electric_kw),
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
