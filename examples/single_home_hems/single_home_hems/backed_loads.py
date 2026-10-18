"""Shared twin/actual load definitions; never count resistance as compressor power."""
import numpy as np

DEFAULT_BACKED_LOADS = ("heat_pump", "water_heater")


def components(results):
    """Accept a raw result row or table; require explicit HVAC component outputs."""
    def field(name):
        if name not in results:
            raise ValueError(f"Missing backed-load component: {name}; use verbosity >= 7")
        return results[name]
    hp = 0
    for use in ("HVAC Heating", "HVAC Cooling"):
        # Main excludes the separately reported ER element; fan is separate.
        hp = hp + field(f"{use} Main Power (kW)") + field(f"{use} Fan Power (kW)")
    return {"heat_pump": hp, "water_heater": field("Water Heating Electric Power (kW)")}


def backed_power(results, selected=DEFAULT_BACKED_LOADS):
    """Named components or explicit end uses. EV/PV/storage cannot back themselves."""
    values = components(results) if "heat_pump" in selected else {}
    total = 0
    for name in selected:
        if name in values:
            value = values[name]
        else:
            column = "Water Heating Electric Power (kW)" if name == "water_heater" else f"{name} Electric Power (kW)"
            if column not in results:
                raise ValueError(f"Unknown or unreported backed load: {name}")
            value = results[column]
        if not np.isfinite(value).all() or np.any(np.asarray(value) < -1e-7):
            raise ValueError(f"Invalid backed-load power for {name}")
        total = total + np.maximum(value, 0)
    return total


def validate_backed_loads(selected):
    """Validate native end uses; reject self-backing EV/storage/total loads."""
    if not isinstance(selected, (list, tuple)) or not selected or len(set(selected)) != len(selected):
        raise ValueError("backed_loads must be a nonempty list of unique components/end uses")
    if any(not isinstance(s, str) or s.startswith("EV") or s in {"PV", "Battery", "Generator", "Total"} for s in selected):
        raise ValueError("Backed loads cannot include EVs, generation, storage, or whole-home totals")
    if ("heat_pump" in selected and set(selected) & {"HVAC", "HVAC Heating", "HVAC Cooling"}) or (
            "water_heater" in selected and "Water Heating" in selected):
        raise ValueError("Overlapping backed-load selections would double-count demand")


def attach_actual_load_cap(dwelling, selected):
    """Evaluate opted-in EVs after non-EV equipment, before the envelope.

    No duplicated thermal step or one-step delay: the callback reads this step's
    realized equipment powers. EVs have no zone heat gains, so moving them after
    ideal HVAC preserves native thermal update dependencies.
    """
    from examples.five_home_block.five_home_block.ev_equipment import BidirectionalEV
    evs = [e for e in dwelling.equipment.values() if isinstance(e, BidirectionalEV)]
    for ev in evs:
        dwelling.sub_simulators.remove(ev)
    pos = dwelling.sub_simulators.index(dwelling.envelope)
    dwelling.sub_simulators[pos:pos] = evs
    def actual():
        row = {}
        for eq in dwelling.equipment.values():
            if eq.end_use == "EV":
                continue
            row.update(eq.generate_results())
            row[f"{eq.end_use} Electric Power (kW)"] = sum(
                e.electric_kw for e in dwelling.equipment.values() if e.end_use == eq.end_use)
        return float(backed_power(row, selected))
    for ev in evs:
        ev.actual_backed_load = actual
