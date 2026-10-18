"""Neighborhood aggregation and power-balance checks."""

from __future__ import annotations

import numpy as np
import pandas as pd


HOME_TO_AGGREGATE = {
    "total_electric_power_kw": "aggregate_total_power_kw",
    "hvac_power_kw": "aggregate_hvac_power_kw",
    "water_heating_power_kw": "aggregate_water_heating_power_kw",
    "ev_power_kw": "aggregate_ev_power_kw",
    "rest_of_home_power_kw": "aggregate_rest_of_home_power_kw",
}


def aggregate_home_results(home_results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Sum aligned standardized home power into neighborhood power."""
    if not home_results:
        raise ValueError("Cannot aggregate an empty collection of home results")
    first_id, first = next(iter(home_results.items()))
    reference_index = first.index
    for home_id, data in home_results.items():
        if not data.index.equals(reference_index):
            raise ValueError(f"Timestamp mismatch between {first_id} and {home_id}")
        missing = set(HOME_TO_AGGREGATE) - set(data.columns)
        if missing:
            raise ValueError(f"{home_id} is missing standardized power fields: {sorted(missing)}")

    aggregate = pd.DataFrame(index=reference_index.copy())
    for home_field, aggregate_field in HOME_TO_AGGREGATE.items():
        aggregate[aggregate_field] = sum(
            data[home_field] for data in home_results.values()
        )
    aggregate.index.name = "timestamp"
    return aggregate


def maximum_power_balance_error(data: pd.DataFrame) -> float:
    """Return maximum absolute total-minus-components power error in kW."""
    balance = data["total_electric_power_kw"] - data[
        ["hvac_power_kw", "water_heating_power_kw", "ev_power_kw", "rest_of_home_power_kw"]
    ].sum(axis=1)
    return float(np.abs(balance).max())


def maximum_aggregate_error(
    home_results: dict[str, pd.DataFrame], aggregate: pd.DataFrame
) -> float:
    """Return the aggregate-total error relative to the direct home sum."""
    direct_sum = sum(data["total_electric_power_kw"] for data in home_results.values())
    return float(np.abs(aggregate["aggregate_total_power_kw"] - direct_sum).max())
