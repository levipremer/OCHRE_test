"""Map version-dependent OCHRE columns to stable neighborhood field names."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd


class OutputMappingError(KeyError):
    """Raised when a required OCHRE result cannot be mapped."""


@dataclass(frozen=True)
class StandardizedResults:
    """Standardized data and the OCHRE columns selected to create it."""

    data: pd.DataFrame
    selected_columns: dict[str, list[str]]


SINGLE_COLUMN_CANDIDATES: dict[str, tuple[str, ...]] = {
    "total_electric_power_kw": (
        "Total Electric Power (kW)",
        "House Electric Power (kW)",
    ),
    "water_heating_power_kw": (
        "Water Heating Electric Power (kW)",
        "Water Heater Electric Power (kW)",
    ),
    "ev_power_kw": (
        "EV Electric Power (kW)",
        "Electric Vehicle Electric Power (kW)",
    ),
    "indoor_temperature_c": (
        "Temperature - Indoor (C)",
        "Indoor Temperature (C)",
    ),
    "ev_soc_fraction": (
        "EV SOC (-)",
        "Electric Vehicle SOC (-)",
    ),
    "water_heater_temperature_c": (
        "Hot Water Average Temperature (C)",
        "Water Heating Average Temperature (C)",
        "Hot Water Outlet Temperature (C)",
    ),
}

OPTIONAL_COLUMN_CANDIDATES: dict[str, tuple[str, ...]] = {
    "heating_setpoint_c": ("HVAC Heating Setpoint (C)",),
    "cooling_setpoint_c": ("HVAC Cooling Setpoint (C)",),
    "water_heater_min_temperature_c": ("Hot Water Minimum Temperature (C)",),
    "water_heater_max_temperature_c": ("Hot Water Maximum Temperature (C)",),
}


def _find_column(columns: pd.Index, candidates: tuple[str, ...]) -> str | None:
    return next((candidate for candidate in candidates if candidate in columns), None)


def _required_column(results: pd.DataFrame, field: str) -> str:
    column = _find_column(results.columns, SINGLE_COLUMN_CANDIDATES[field])
    if column is None:
        raise OutputMappingError(
            f"Cannot map required field {field!r}. Tried "
            f"{list(SINGLE_COLUMN_CANDIDATES[field])}. Available OCHRE columns: "
            f"{results.columns.tolist()}"
        )
    return column


def _map_hvac_power(results: pd.DataFrame) -> tuple[pd.Series, list[str]]:
    combined = _find_column(results.columns, ("HVAC Electric Power (kW)",))
    if combined:
        return pd.to_numeric(results[combined], errors="raise"), [combined]

    components = [
        column
        for column in (
            "HVAC Heating Electric Power (kW)",
            "HVAC Cooling Electric Power (kW)",
        )
        if column in results.columns
    ]
    if not components:
        raise OutputMappingError(
            "Cannot map HVAC power. Expected a combined HVAC column or heating/cooling "
            f"components. Available OCHRE columns: {results.columns.tolist()}"
        )
    return results.loc[:, components].apply(pd.to_numeric, errors="raise").sum(axis=1), components


def standardize_results(
    results: pd.DataFrame,
    home_id: str,
    numerical_tolerance_kw: float = 1e-6,
) -> StandardizedResults:
    """Convert raw OCHRE results into stable fields used by this example."""
    if not isinstance(results.index, pd.DatetimeIndex):
        raise OutputMappingError(f"{home_id}: OCHRE results must use a DatetimeIndex")
    if results.index.has_duplicates:
        raise OutputMappingError(f"{home_id}: OCHRE results contain duplicate timestamps")

    standardized = pd.DataFrame(index=results.index.copy())
    selected: dict[str, list[str]] = {}
    for field in SINGLE_COLUMN_CANDIDATES:
        column = _required_column(results, field)
        standardized[field] = pd.to_numeric(results[column], errors="raise")
        selected[field] = [column]

    hvac, hvac_columns = _map_hvac_power(results)
    standardized["hvac_power_kw"] = hvac
    selected["hvac_power_kw"] = hvac_columns

    for field, candidates in OPTIONAL_COLUMN_CANDIDATES.items():
        column = _find_column(results.columns, candidates)
        if column is not None:
            standardized[field] = pd.to_numeric(results[column], errors="raise")
            selected[field] = [column]

    standardized["rest_of_home_power_kw"] = (
        standardized["total_electric_power_kw"]
        - standardized["hvac_power_kw"]
        - standardized["water_heating_power_kw"]
        - standardized["ev_power_kw"]
    )
    tiny_negative = standardized["rest_of_home_power_kw"].between(
        -numerical_tolerance_kw, 0, inclusive="left"
    )
    standardized.loc[tiny_negative, "rest_of_home_power_kw"] = 0.0
    material_negative = standardized["rest_of_home_power_kw"] < -numerical_tolerance_kw
    if material_negative.any():
        minimum = standardized.loc[material_negative, "rest_of_home_power_kw"].min()
        warnings.warn(
            f"{home_id}: materially negative rest-of-home power ({minimum:.6g} kW). "
            "Check output mapping and sign conventions.",
            RuntimeWarning,
            stacklevel=2,
        )
    selected["rest_of_home_power_kw"] = [
        "derived: total - HVAC - water heating - EV"
    ]
    standardized.index.name = "timestamp"
    return StandardizedResults(standardized, selected)
