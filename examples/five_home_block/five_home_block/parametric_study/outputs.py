"""Profile standardization and exact neighborhood reconstruction helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import ResolvedSeason


ELECTRIC_COMPONENT_COLUMNS = (
    "hvac_electric_power_kw",
    "water_heating_electric_power_kw",
    "cooking_electric_power_kw",
    "ev_electric_power_kw",
    "rest_of_home_electric_power_kw",
)

AGGREGATABLE_COLUMNS = (
    "total_electric_power_kw",
    "total_reactive_power_kvar",
    "total_gas_power_therms_per_hour",
    "hvac_electric_power_kw",
    "hvac_gas_power_therms_per_hour",
    "water_heating_electric_power_kw",
    "water_heating_gas_power_therms_per_hour",
    "cooking_electric_power_kw",
    "cooking_gas_power_therms_per_hour",
    "ev_electric_power_kw",
    "rest_of_home_electric_power_kw",
    "rest_of_home_gas_power_therms_per_hour",
)


def _numeric_series(raw: pd.DataFrame, column: str) -> pd.Series:
    if column not in raw:
        return pd.Series(0.0, index=raw.index)
    return pd.to_numeric(raw[column], errors="raise")


def standardize_profile(raw: pd.DataFrame, period: ResolvedSeason) -> pd.DataFrame:
    """Retain transformer-agnostic electric, reactive, gas, and state data."""
    if not isinstance(raw.index, pd.DatetimeIndex) or raw.index.has_duplicates:
        raise ValueError("OCHRE profile requires a unique DatetimeIndex")
    data = pd.DataFrame(index=raw.index.copy())
    data["total_electric_power_kw"] = _numeric_series(raw, "Total Electric Power (kW)")
    data["total_reactive_power_kvar"] = _numeric_series(
        raw, "Total Reactive Power (kVAR)"
    )
    data["total_gas_power_therms_per_hour"] = _numeric_series(
        raw, "Total Gas Power (therms/hour)"
    )
    data["hvac_electric_power_kw"] = _numeric_series(
        raw, "HVAC Heating Electric Power (kW)"
    ) + _numeric_series(raw, "HVAC Cooling Electric Power (kW)")
    data["hvac_gas_power_therms_per_hour"] = _numeric_series(
        raw, "HVAC Heating Gas Power (therms/hour)"
    )
    data["water_heating_electric_power_kw"] = _numeric_series(
        raw, "Water Heating Electric Power (kW)"
    )
    data["water_heating_gas_power_therms_per_hour"] = _numeric_series(
        raw, "Water Heating Gas Power (therms/hour)"
    )
    data["cooking_electric_power_kw"] = _numeric_series(
        raw, "Cooking Range Electric Power (kW)"
    )
    data["cooking_gas_power_therms_per_hour"] = _numeric_series(
        raw, "Cooking Range Gas Power (therms/hour)"
    )
    data["ev_electric_power_kw"] = _numeric_series(raw, "EV Electric Power (kW)")
    data["rest_of_home_electric_power_kw"] = data["total_electric_power_kw"] - data[
        [
            "hvac_electric_power_kw",
            "water_heating_electric_power_kw",
            "cooking_electric_power_kw",
            "ev_electric_power_kw",
        ]
    ].sum(axis=1)
    data["rest_of_home_gas_power_therms_per_hour"] = data[
        "total_gas_power_therms_per_hour"
    ] - data[
        [
            "hvac_gas_power_therms_per_hour",
            "water_heating_gas_power_therms_per_hour",
            "cooking_gas_power_therms_per_hour",
        ]
    ].sum(axis=1)
    for column in (
        "rest_of_home_electric_power_kw",
        "rest_of_home_gas_power_therms_per_hour",
    ):
        tiny = data[column].between(-1e-8, 0, inclusive="left")
        data.loc[tiny, column] = 0.0

    optional = {
        "indoor_temperature_c": "Temperature - Indoor (C)",
        "water_heater_average_temperature_c": "Hot Water Average Temperature (C)",
        "water_heater_minimum_temperature_c": "Hot Water Minimum Temperature (C)",
        "water_heater_maximum_temperature_c": "Hot Water Maximum Temperature (C)",
        "ev_soc_fraction": "EV SOC (-)",
        "outdoor_temperature_c": "Temperature - Outdoor (C)",
    }
    for field, source in optional.items():
        if source in raw:
            data[field] = pd.to_numeric(raw[source], errors="raise")

    report_start = pd.Timestamp(period.report_start)
    report_end = pd.Timestamp(period.report_end)
    if data.index.tz is not None:
        report_start = report_start.tz_localize(data.index.tz)
        report_end = report_end.tz_localize(data.index.tz)
    data["is_reporting_period"] = (data.index >= report_start) & (data.index < report_end)
    data.index.name = "timestamp"
    if not np.isfinite(data.drop(columns="is_reporting_period").to_numpy(float)).all():
        raise ValueError("Standardized profile contains nonfinite values")
    return data


def validate_profile(
    data: pd.DataFrame,
    period: ResolvedSeason,
    time_step,
    maximum_home_power_kw: float,
) -> None:
    expected_rows = period.duration // time_step
    if len(data) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} profile rows, got {len(data)}")
    expected_reporting = (period.report_end - period.report_start) // time_step
    if int(data["is_reporting_period"].sum()) != expected_reporting:
        raise RuntimeError("Reporting-period flag does not match the selected season")
    if data["total_electric_power_kw"].max() > maximum_home_power_kw:
        raise RuntimeError("Profile exceeds the configured per-home plausibility limit")
    electric_balance = data[list(ELECTRIC_COMPONENT_COLUMNS)].sum(axis=1)
    if (data["total_electric_power_kw"] - electric_balance).abs().max() > 1e-6:
        raise RuntimeError("Profile electric-power components do not balance")


def reconstruct_neighborhood_profile(
    scenario: pd.Series | dict[str, object], profiles_directory: Path
) -> pd.DataFrame:
    """Reconstruct one scenario exactly from its five cached profile IDs."""
    profile_ids = [scenario[f"home_{number:02d}_profile_id"] for number in range(1, 6)]
    profiles = [
        pd.read_parquet(profiles_directory / f"{profile_id}.parquet")
        for profile_id in profile_ids
    ]
    reference_index = profiles[0].index
    if any(not profile.index.equals(reference_index) for profile in profiles[1:]):
        raise ValueError("Cached profiles are not timestamp aligned")
    aggregate = pd.DataFrame(index=reference_index.copy())
    for column in AGGREGATABLE_COLUMNS:
        aggregate[column] = sum(profile[column] for profile in profiles)
    aggregate["is_reporting_period"] = profiles[0]["is_reporting_period"].astype(bool)
    aggregate.index.name = "timestamp"
    return aggregate
