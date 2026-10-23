"""Evaluation-window EV metrics from realized plant states and damage ledgers.

Warmup supplies the boundary state, not scored costs. Optimization predictions
never update these metrics; baseline, twin and advanced are evaluated separately.
"""
import numpy as np
import pandas as pd
from examples.five_home_block.five_home_block.ev_degradation import BatteryDegradation


def ev_operating_state(raw, name):
    """Actual AC-power state; availability wins over idle at zero away power."""
    power = raw[f"{name} Electric Power (kW)"]
    state = pd.Series("idle", index=raw.index)
    state.loc[power > 1e-7] = "charging"
    state.loc[power < -1e-7] = "discharging"
    parked = raw.get(f"{name} Parked", raw.get(f"{name} Parked (-)"))
    if parked is None:
        parked = raw[f"{name} Mode"].ne("Away")
    state.loc[~parked.astype(bool)] = "away"
    return state

def ev_metrics(raw, home, prices, step_hours, controllers=(), *, previous=None, mobility=None):
    """Evaluation-only metrics; previous is the final warm-up plant row, if any."""
    rows = []
    by_name = {c.equipment_name: c for c in controllers}
    for ev in home.evs:
        name = f"EV_{ev.vehicle_id}"
        power = raw[f"{name} Electric Power (kW)"]
        soc = raw[f"{name} SOC (-)"]
        end_soc = raw.get(f"{name} End SOC (-)", soc)
        unmet = raw[f"{name} Unmet Load (kWh)"]
        controller = by_name.get(name)
        attempts = [a for a in controller.solve_history if raw.index[0] <= a["timestamp"] <= raw.index[-1]] if controller else []
        actions = [a for a in controller.action_history if raw.index[0] <= a["timestamp"] <= raw.index[-1]] if controller else []
        seconds = sum(a["solve_seconds"] for a in attempts)
        positive, negative = power.gt(1e-7), power.lt(-1e-7)
        initial_soc = float(previous[f"{name} End SOC (-)"]) if previous is not None else ev.initial_soc_fraction
        previous_power = float(previous[f"{name} Electric Power (kW)"]) if previous is not None else 0.0
        charge_starts = int((positive & ~positive.shift(fill_value=previous_power > 1e-7)).sum())
        discharge_starts = int((negative & ~negative.shift(fill_value=previous_power < -1e-7)).sum())
        row = dict(vehicle_id=ev.vehicle_id, charge_energy_kwh=power.clip(lower=0).sum()*step_hours,
                   v2h_energy_kwh=-power.clip(upper=0).sum()*step_hours,
                   net_energy_kwh=power.sum()*step_hours, average_power_kw=power.mean(),
                   peak_charge_kw=power.max(), peak_v2h_kw=max(0, -power.min()),
                   operating_cost_usd=(power*prices).sum()*step_hours,
                   initial_soc=initial_soc, configured_initial_soc=ev.initial_soc_fraction, capacity_kwh=ev.capacity_kwh,
                   initial_stored_energy_kwh=initial_soc*ev.capacity_kwh,
                   final_soc=float(end_soc.iloc[-1]), final_stored_energy_kwh=float(end_soc.iloc[-1])*ev.capacity_kwh,
                   stored_energy_change_kwh=(float(end_soc.iloc[-1])-initial_soc)*ev.capacity_kwh,
                   insufficient_soc_events=int(unmet.gt(1e-7).sum()),
                   insufficient_soc_kwh=unmet.sum(),
                   charge_starts=charge_starts, discharge_starts=discharge_starts,
                   cycle_count=charge_starts+discharge_starts,
                   solve_attempts=len(attempts), failed_solves=sum(not a["successful"] for a in attempts),
                   failed_solve_fraction=sum(not a["successful"] for a in attempts)/len(attempts) if attempts else 0.0,
                   fallback_hours=sum(bool(a.get("fallback", False)) for a in actions)*step_hours,
                   mean_solve_seconds=seconds/len(attempts) if attempts else 0.0,
                   p95_solve_seconds=float(np.quantile([a["solve_seconds"] for a in attempts], .95)) if attempts else 0.0,
                   max_solve_seconds=max((a["solve_seconds"] for a in attempts), default=0.0),
                   max_ocp_build_and_solve_seconds=max((a["ocp_seconds"] for a in attempts), default=0.0),
                   solve_seconds_per_sim_timestep=seconds/len(raw),
                   ocp_seconds_per_sim_timestep=sum(a["ocp_seconds"] for a in attempts)/len(raw),
                   mean_ocp_build_and_solve_seconds=(sum(a["ocp_seconds"] for a in attempts)/len(attempts)
                                                     if attempts else 0.0))
        if ev.hems:
            # Keep whole-window comfort diagnostics for compatibility, but also
            # report the departure-only quantity used by the revised OCP.
            if mobility is not None and ev.vehicle_id in mobility:
                departures = pd.DatetimeIndex(mobility[ev.vehicle_id].trips.departure.unique())
                departures = departures[(departures >= raw.index[0]) &
                                        (departures < raw.index[-1]+pd.Timedelta(hours=step_hours))]
                # New outputs explicitly retain the state BEFORE departure
                # depletion. Old arrival-accounted outputs keep their meaning.
                departure_soc = raw.get(f"{name} Pre-departure SOC (-)", soc).reindex(departures)
                if departure_soc.isna().any():
                    raise ValueError(f"{name}: departure timestamps must align with saved plant states")
                shortfall = (ev.hems.comfort_soc_min-departure_soc).clip(lower=0)
                row.update(departure_count=len(departures),
                           departure_comfort_shortfall_events=int((shortfall > 1e-7).sum()),
                           mean_departure_comfort_shortfall=float(shortfall.mean()) if len(shortfall) else 0.,
                           max_departure_comfort_shortfall=float(shortfall.max()) if len(shortfall) else 0.)
            row.update(health_low_soc_hours=(ev.hems.health_soc_min-end_soc).clip(lower=0).sum()*step_hours,
                       away_command_rejected_steps=int(raw.get(f"{name} Away Command Rejected", pd.Series(dtype=float)).sum()),
                       health_high_soc_hours=(end_soc-ev.hems.health_soc_max).clip(lower=0).sum()*step_hours,
                       comfort_soc_shortfall_hours=(ev.hems.comfort_soc_min-end_soc).clip(lower=0).sum()*step_hours,
                       max_comfort_soc_shortfall=float((ev.hems.comfort_soc_min-end_soc).clip(lower=0).max()),
                       soc_saturation_steps=int(raw.get(f"{name} SOC Saturation", pd.Series(dtype=float)).sum()),
                       soc_clipped_kwh=raw.get(f"{name} SOC Clipped Energy (kWh)", pd.Series(dtype=float)).sum(),
                       trip_unserved_kwh=raw.get(f"{name} Trip Unserved Energy (kWh)", pd.Series(dtype=float)).sum(),
                       safety_overrides=int(raw[f"{name} Safety Override"].sum()),
                       projection_overrides=sum(a["projection_override"] for a in controller.action_history
                                                if raw.index[0] <= a["timestamp"] <= raw.index[-1]) if controller else 0)
            # Include pre/post plant-step states: charging and trip impulses
            # must not cancel and hide an extremum. Unserved driving is excluded.
            path = np.r_[initial_soc, np.column_stack([soc, end_soc]).ravel()]
            battery = BatteryDegradation(ev.capacity_kwh, ev.hems)
            if f"{name} Damage Increment (-)" in raw:
                # Score increments of the continuous ledger, including open
                # cycles carried across warmup. Never recount a clipped trace.
                costs = {f"{method}_degradation_cost_usd": float(raw[f"{name} {column} Degradation Cost ($)"].sum())
                         for method, column in (("reference", "Reference"), ("rainflow", "Rainflow"),
                                                ("linear", "Throughput"))}
                costs["battery_throughput_kwh"] = float(raw[f"{name} Battery Throughput (kWh)"].sum())
                initial_energy = float(raw[f"{name} Start Stored Energy (kWh)"].iloc[0])
                final_energy = float(raw[f"{name} End Stored Energy (kWh)"].iloc[-1])
                initial_soh = float(raw[f"{name} Start SOH (-)"].iloc[0])
                final_soh = float(raw[f"{name} End SOH (-)"].iloc[-1])
                final_damage = float(raw[f"{name} End Damage (-)"].iloc[-1])
                row.update(degradation_accounting="continuous_dynamic_soh",
                           configured_initial_soh=ev.hems.initial_soh,
                           initial_soh=initial_soh, final_soh=final_soh,
                           soh_loss=initial_soh-final_soh,
                           initial_capacity_kwh=ev.capacity_kwh*initial_soh,
                           final_capacity_kwh=ev.capacity_kwh*final_soh,
                           initial_stored_energy_kwh=initial_energy,
                           final_stored_energy_kwh=final_energy,
                           stored_energy_change_kwh=final_energy-initial_energy,
                           initial_damage=(1-initial_soh)/battery.calibration_fade,
                           final_damage=final_damage,
                           damage_increment=float(raw[f"{name} Damage Increment (-)"].sum()),
                           remaining_economic_damage_budget=battery.allowed_damage-final_damage,
                           economic_eol_reached=bool(raw[f"{name} Economic EOL Reached"].any()),
                           fade_energy_loss_kwh=float(raw[f"{name} Fade Energy Loss (kWh)"].sum()),
                           equivalent_full_cycles_current_capacity=float(
                               raw[f"{name} Throughput Damage Increment (-)"].sum())*
                               ev.hems.reference_cycle_count*ev.hems.reference_cycle_depth)
            else:
                # Explicit compatibility for archived static-SOH outputs only.
                costs = battery.costs(path)
                row["degradation_accounting"] = "legacy_static_soh"
            row.update(costs)
            row["equivalent_full_cycles"] = costs["battery_throughput_kwh"]/(2*ev.capacity_kwh)
            method = "linear" if ev.hems.degradation_method == "linear_throughput" else "rainflow"
            row.update(degradation_method=ev.hems.degradation_method,
                       estimated_degradation_cost_usd=(np.nan if ev.hems.degradation_method == "none"
                                                      else costs[f"{method}_degradation_cost_usd"]),
                       reference_cycle_life=battery.reference_cycle_life,
                       calibration_cycle_count=ev.hems.reference_cycle_count,
                       calibration_end_of_life_soh=ev.hems.calibration_end_of_life_soh,
                       economic_end_of_life_soh=ev.hems.end_of_life_soh,
                       replacement_cost_usd=battery.replacement_usd,
                       residual_value_usd=battery.residual_usd,
                       depreciable_cost_usd=battery.depreciable_usd)
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
        "vehicle_id", "failed_solves", "solve_attempts", "mean_solve_seconds", "final_soc"])
