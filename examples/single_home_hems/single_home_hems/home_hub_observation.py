"""The electrical measurements available to an EV-only Home Hub controller.

OCHRE retains its detailed equipment outputs for plant physics and reporting.
This projection represents a single Hub meter and a separate EV meter; it is
the only completed plant status passed to the EV-only controller or predictor.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


HUB_POWER_KEY = "Home Hub Electric Power (kW)"
TOTAL_POWER_KEY = "Total Electric Power (kW)"
EV_OBSERVATION_SUFFIXES = (
    "Electric Power (kW)",
    "SOC (-)",
    "End SOC (-)",
    "SOH (-)",
    "Capacity (kWh)",
    "Parked",
)
HUB_COMPONENT_POWER_KEYS = (
    # Backup electric resistance is wired outside this Home Hub. OCHRE's
    # Heating Electric total includes it; main plus fan does not.
    "HVAC Heating Main Power (kW)",
    "HVAC Heating Fan Power (kW)",
    "HVAC Cooling Electric Power (kW)",
    "Water Heating Electric Power (kW)",
    "DC House Hub Other Electric Power (kW)",
)


def observation_keys(ev_name: str, *, include_total_home: bool = False) -> tuple[str, ...]:
    return ((HUB_POWER_KEY, TOTAL_POWER_KEY) if include_total_home else (HUB_POWER_KEY,)) + (
        *(f"{ev_name} {suffix}" for suffix in EV_OBSERVATION_SUFFIXES),
    )


def project_ev_only_hub_observation(
    plant_status: Mapping[str, Any], ev_name: str, *, include_total_home: bool = False,
) -> dict[str, Any]:
    """Combine the live Hub circuits, including signed EV AC power, at the meter."""
    keys = observation_keys(ev_name, include_total_home=include_total_home)
    ev_keys = observation_keys(ev_name)[1:]
    needed = (*HUB_COMPONENT_POWER_KEYS, *ev_keys)
    if include_total_home:
        needed = (*needed, TOTAL_POWER_KEY)
    missing = set(needed) - set(plant_status)
    if missing:
        raise ValueError(f"EV-only Home Hub plant status lacks {sorted(missing)}")
    component_power = [float(plant_status[key]) for key in HUB_COMPONENT_POWER_KEYS]
    ev_power = float(plant_status[f"{ev_name} Electric Power (kW)"])
    if not np.isfinite([*component_power, ev_power]).all():
        raise ValueError("EV-only Home Hub meter inputs must be finite")
    if any(power < -1e-7 for power in component_power):
        raise ValueError("EV-only Home Hub non-EV circuits cannot have negative power")
    hub_power = sum(component_power) + ev_power
    if hub_power < -1e-7:
        raise ValueError("EV-only Home Hub net demand cannot be negative")
    projected = {HUB_POWER_KEY: hub_power,
                 **{key: plant_status[key] for key in ev_keys}}
    if include_total_home:
        total = float(plant_status[TOTAL_POWER_KEY])
        if not np.isfinite(total) or total < hub_power - 1e-7:
            raise ValueError("Completed total-home meter is below corrected Home Hub power")
        projected[TOTAL_POWER_KEY] = total
    if set(projected) != set(keys):
        raise AssertionError("EV-only aggregate meter projection changed schema")
    return projected


def validate_ev_only_hub_observation(
    observation: Mapping[str, Any], ev_name: str, *, include_total_home: bool = False,
) -> None:
    """Reject hidden equipment channels at the controller boundary."""
    expected = set(observation_keys(ev_name, include_total_home=include_total_home))
    actual = set(observation)
    if actual != expected:
        raise ValueError(
            "EV-only Home Hub observation must contain only the Hub meter and "
            f"this EV's meter/state; missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    hub_power = float(observation[HUB_POWER_KEY])
    ev_power = float(observation[f"{ev_name} Electric Power (kW)"])
    if not np.isfinite([hub_power, ev_power]).all() or hub_power < -1e-7:
        raise ValueError("EV-only Home Hub meter power must be finite and nonnegative")
    if include_total_home:
        total = float(observation[TOTAL_POWER_KEY])
        if not np.isfinite(total) or total < hub_power - 1e-7:
            raise ValueError("EV-only total-home meter must be finite and at least corrected Hub power")
