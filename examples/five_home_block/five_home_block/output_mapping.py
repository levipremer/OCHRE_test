"""Map version-dependent OCHRE columns to stable neighborhood field names."""

from __future__ import annotations

import re
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
    "water_heater_temperature_c": (
        "Hot Water Average Temperature (C)",
        "Water Heating Average Temperature (C)",
        "Hot Water Outlet Temperature (C)",
    ),
}

LEGACY_EV_SOC_COLUMNS = (
    "EV SOC (-)",
    "Electric Vehicle SOC (-)",
)

PER_EV_RESULT_FIELDS = {
    "Electric Power (kW)": "power_kw",
    "SOC (-)": "soc_fraction",
    "Unmet Load (kWh)": "unmet_load_kwh",
}

OPTIONAL_COLUMN_CANDIDATES: dict[str, tuple[str, ...]] = {
    "heating_setpoint_c": ("HVAC Heating Setpoint (C)",),
    "cooling_setpoint_c": ("HVAC Cooling Setpoint (C)",),
    "water_heater_min_temperature_c": ("Hot Water Minimum Temperature (C)",),
    "water_heater_max_temperature_c": ("Hot Water Maximum Temperature (C)",),
}


def _find_column(columns: pd.Index, candidates: tuple[str, ...]) -> str | None:
    return next((candidate for candidate in candidates if candidate in columns), None)


def find_per_ev_columns(columns: pd.Index, suffix: str) -> dict[str, str]:
    """Map normalized vehicle IDs to uniquely prefixed OCHRE result columns."""
    ending = f" {suffix}"
    matches: dict[str, str] = {}
    for column in columns:
        if not isinstance(column, str) or not column.startswith("EV_"):
            continue
        if not column.endswith(ending):
            continue
        vehicle_id = column[len("EV_") : -len(ending)]
        normalized = re.sub(r"[^a-z0-9]+", "_", vehicle_id.lower()).strip("_")
        if not normalized:
            raise OutputMappingError(f"Invalid per-EV result column: {column!r}")
        previous = matches.get(normalized)
        if previous is not None:
            raise OutputMappingError(
                f"Per-EV result columns {previous!r} and {column!r} normalize "
                f"to the same vehicle ID {normalized!r}"
            )
        matches[normalized] = column
    return matches


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
    *, expected_ev_count: int | None = None,
    resistance_outside_hub: bool = False,
) -> StandardizedResults:
    """Convert raw OCHRE results into stable fields used by this example."""
    if not isinstance(results.index, pd.DatetimeIndex):
        raise OutputMappingError(f"{home_id}: OCHRE results must use a DatetimeIndex")
    if results.index.has_duplicates:
        raise OutputMappingError(f"{home_id}: OCHRE results contain duplicate timestamps")

    standardized = pd.DataFrame(index=results.index.copy())
    selected: dict[str, list[str]] = {}
    for field in SINGLE_COLUMN_CANDIDATES:
        if field == "ev_power_kw" and expected_ev_count == 0:
            if find_per_ev_columns(results.columns, "SOC (-)"):
                raise OutputMappingError("EV results conflict with a home configured without EVs")
            aggregate = _find_column(results.columns, SINGLE_COLUMN_CANDIDATES[field])
            if aggregate is not None and results[aggregate].abs().max() > numerical_tolerance_kw:
                raise OutputMappingError("Nonzero EV power in a home configured without EVs")
            standardized[field] = 0.0
            selected[field] = ["configured: no EV equipment"]
            continue
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

    per_ev_fields: dict[str, list[str]] = {
        standardized_suffix: []
        for standardized_suffix in PER_EV_RESULT_FIELDS.values()
    }
    for result_suffix, standardized_suffix in PER_EV_RESULT_FIELDS.items():
        for vehicle_id, source in find_per_ev_columns(results.columns, result_suffix).items():
            field = f"ev_{vehicle_id}_{standardized_suffix}"
            standardized[field] = pd.to_numeric(results[source], errors="raise")
            selected[field] = [source]
            per_ev_fields[standardized_suffix].append(field)

    per_ev_soc_fields = per_ev_fields["soc_fraction"]
    if len(per_ev_soc_fields) == 1:
        standardized["ev_soc_fraction"] = standardized[per_ev_soc_fields[0]]
        selected["ev_soc_fraction"] = [
            f"alias: {selected[per_ev_soc_fields[0]][0]}"
        ]
    elif not per_ev_soc_fields and expected_ev_count != 0:
        legacy_soc = _find_column(results.columns, LEGACY_EV_SOC_COLUMNS)
        if legacy_soc is None:
            raise OutputMappingError(
                "Cannot map EV SOC. Expected a uniquely prefixed per-EV SOC column "
                f"or one of {list(LEGACY_EV_SOC_COLUMNS)}. Available OCHRE columns: "
                f"{results.columns.tolist()}"
            )
        standardized["ev_soc_fraction"] = pd.to_numeric(
            results[legacy_soc], errors="raise"
        )
        selected["ev_soc_fraction"] = [legacy_soc]

    per_ev_power_fields = per_ev_fields["power_kw"]
    if per_ev_power_fields:
        per_ev_total = standardized[per_ev_power_fields].sum(axis=1)
        maximum_error = (standardized["ev_power_kw"] - per_ev_total).abs().max()
        if maximum_error > numerical_tolerance_kw:
            raise OutputMappingError(
                f"{home_id}: aggregate EV power differs from the sum of per-EV "
                f"power by up to {maximum_error:.6g} kW"
            )

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
    split_columns = {
        "hub_other_power_kw": "DC House Hub Other Electric Power (kW)",
        "outside_hub_power_kw": "DC House Outside Hub Electric Power (kW)",
    }
    present = {field: column for field, column in split_columns.items() if column in results}
    if present and len(present) != len(split_columns):
        raise OutputMappingError(
            f"{home_id}: Home Hub replay must report both split loads; found {list(present)}"
        )
    if present:
        for field, column in split_columns.items():
            standardized[field] = pd.to_numeric(results[column], errors="raise")
            selected[field] = [column]
        active_columns = [
            "DC House Hub Other Replay Active (-)",
            "DC House Outside Hub Replay Active (-)",
        ]
        if all(column in results for column in active_columns):
            hub_active = pd.to_numeric(results[active_columns[0]], errors="raise").astype(bool)
            outside_active = pd.to_numeric(results[active_columns[1]], errors="raise").astype(bool)
            if not hub_active.equals(outside_active):
                raise OutputMappingError(f"{home_id}: Home Hub split replay activity disagrees")
            active = hub_active
        else:
            active = pd.Series(True, index=results.index)
        split_total = standardized["hub_other_power_kw"] + standardized["outside_hub_power_kw"]
        error = (standardized.loc[active, "rest_of_home_power_kw"] - split_total.loc[active]).abs().max()
        if pd.notna(error) and (not np.isfinite(error) or error > numerical_tolerance_kw):
            raise OutputMappingError(
                f"{home_id}: Home Hub split disagrees with derived rest-of-home power "
                f"by up to {error:.6g} kW"
            )
        if resistance_outside_hub:
            resistance_key = "HVAC Heating ER Power (kW)"
            if resistance_key not in results:
                raise OutputMappingError(
                    f"{home_id}: resistance-outside-Hub reporting requires {resistance_key}"
                )
            resistance = pd.to_numeric(results[resistance_key], errors="raise")
            if (not np.isfinite(resistance.to_numpy(float)).all()
                    or resistance.lt(-numerical_tolerance_kw).any()):
                raise OutputMappingError(f"{home_id}: invalid HVAC resistance power")
            standardized["hvac_resistance_power_kw"] = resistance
            selected["hvac_resistance_power_kw"] = [resistance_key]
            standardized["outside_hub_with_resistance_power_kw"] = (
                standardized["outside_hub_power_kw"] + resistance
            )
            selected["outside_hub_with_resistance_power_kw"] = [
                "derived: fixed Outside-Hub + HVAC resistance"
            ]
            standardized["home_hub_non_ev_power_kw"] = (
                standardized["hvac_power_kw"] - resistance
                + standardized["water_heating_power_kw"]
                + standardized["hub_other_power_kw"]
            )
            selected["home_hub_non_ev_power_kw"] = [
                "derived: HVAC minus resistance + water heating + Hub-other"
            ]
            whole_home_error = (
                standardized["total_electric_power_kw"]
                - standardized["home_hub_non_ev_power_kw"]
                - standardized["ev_power_kw"]
                - standardized["outside_hub_with_resistance_power_kw"]
            ).abs().max()
            if pd.notna(whole_home_error) and whole_home_error > numerical_tolerance_kw:
                raise OutputMappingError(
                    f"{home_id}: resistance-outside-Hub whole-home balance differs "
                    f"by up to {whole_home_error:.6g} kW"
                )
        else:
            standardized["home_hub_non_ev_power_kw"] = (
                standardized["hvac_power_kw"]
                + standardized["water_heating_power_kw"]
                + standardized["hub_other_power_kw"]
            )
            selected["home_hub_non_ev_power_kw"] = [
                "derived: HVAC + water heating + Hub-other"
            ]
    elif resistance_outside_hub:
        raise OutputMappingError(
            f"{home_id}: resistance-outside-Hub reporting requires split replay loads"
        )
    standardized.index.name = "timestamp"
    return StandardizedResults(standardized, selected)
