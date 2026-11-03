"""Passive EV-study summaries from explicitly selected raw OCHRE minute rows.

The caller selects the evaluation window. Departure energy sufficiency and
actual unapplied arrival-trip energy are distinct physical records. Chemical
inventories use the saved energy ledger and measured faded capacities.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ev_study_metrics(
    raw: pd.DataFrame,
    ev_name: str,
    *,
    health_soc_min: float,
    health_soc_max: float,
    comfort_soc_min: float,
    step_hours: float = 1 / 60,
) -> dict:
    """Summarize one EV's complete, already-filtered evaluation minute rows.

    SOC-health durations and integrals use each minute's END state and are
    descriptive realized metrics, not reconstructed optimization objectives.
    Departure comfort uses the departure minute's post-boundary/pre-trip SOC.
    """
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise ValueError("EV study metrics require nonempty raw evaluation rows")
    if not isinstance(raw.index, pd.DatetimeIndex) or raw.index.hasnans:
        raise ValueError("EV study metrics require a DatetimeIndex")
    if raw.index.has_duplicates or not raw.index.is_monotonic_increasing:
        raise ValueError("EV study metrics need unique increasing timestamps")
    if not np.isfinite(step_hours) or step_hours <= 0:
        raise ValueError("EV study metric step_hours must be finite and positive")
    step = pd.Timedelta(hours=step_hours)
    if len(raw) > 1 and (np.diff(raw.index.asi8) != step.value).any():
        raise ValueError("EV study evaluation rows must cover every consecutive timestep")
    if not (np.isfinite([health_soc_min, health_soc_max, comfort_soc_min]).all()
            and 0 <= health_soc_min <= health_soc_max <= 1 and 0 <= comfort_soc_min <= 1):
        raise ValueError("EV study SOC bounds must satisfy 0 <= health min <= max <= 1")

    required = (
        "SOC (-)", "End SOC (-)", "Start SOH (-)", "End SOH (-)",
        "Start Stored Energy (kWh)", "End Stored Energy (kWh)",
        "Capacity (kWh)", "End Capacity (kWh)", "Unmet Load (kWh)",
        "Trip Unserved Energy (kWh)", "Departures",
    )
    missing = [f"{ev_name} {suffix}" for suffix in required
               if f"{ev_name} {suffix}" not in raw]
    if missing:
        raise ValueError(f"EV study raw results lack {missing}")

    def values(suffix):
        series = pd.to_numeric(raw[f"{ev_name} {suffix}"], errors="raise").astype(float)
        if not np.isfinite(series.to_numpy()).all():
            raise ValueError(f"EV study raw {suffix} must be finite")
        return series

    data = {suffix: values(suffix) for suffix in required}
    soc, end_soc = data["SOC (-)"], data["End SOC (-)"]
    start_soh, end_soh = data["Start SOH (-)"], data["End SOH (-)"]
    start_energy = float(data["Start Stored Energy (kWh)"].iloc[0])
    end_energy = float(data["End Stored Energy (kWh)"].iloc[-1])
    end_capacity = data["End Capacity (kWh)"]
    if ((start_soh <= 0).any() or (end_soh <= 0).any()
            or (data["Capacity (kWh)"] <= 0).any() or (end_capacity <= 0).any()):
        raise ValueError("EV study measured SOH and capacity must be positive")
    nominal_capacity = end_capacity / end_soh
    if not np.allclose(nominal_capacity, nominal_capacity.iloc[0], rtol=1e-8, atol=1e-8):
        raise ValueError("EV study nominal capacity inferred from measured SOH must stay fixed")
    initial_capacity = float(nominal_capacity.iloc[0] * start_soh.iloc[0])
    initial_soc = start_energy / initial_capacity
    departure_shortfall = data["Unmet Load (kWh)"]
    departures = data["Departures"]
    if ((departure_shortfall < -1e-8).any() or (departures < 0).any()
            or not np.equal(departures, np.round(departures)).all()):
        raise ValueError("EV study departure counts and energy shortfalls must be nonnegative")
    arrival_unapplied = data["Trip Unserved Energy (kWh)"]
    low, high = (health_soc_min - end_soc).clip(lower=0), (end_soc - health_soc_max).clip(lower=0)
    departure_comfort = (comfort_soc_min - soc).clip(lower=0) * departures
    departure_count = int(departures.sum())
    duration = len(raw) * step_hours
    result = {
        "evaluation_start": raw.index[0].isoformat(),
        "evaluation_end_exclusive": (raw.index[-1] + step).isoformat(),
        "evaluation_duration_hours": float(duration), "evaluation_timestep_hours": float(step_hours),
        "departure_count": departure_count,
        "departure_energy_shortfall_kwh": float(departure_shortfall.clip(lower=0).sum()),
        "departure_minutes_with_energy_shortfall_count": int(departure_shortfall.gt(1e-9).sum()),
        "arrival_trip_unserved_energy_kwh": float(arrival_unapplied.clip(lower=0).sum()),
        "arrival_minutes_with_unserved_trip_energy_count": int(arrival_unapplied.gt(1e-9).sum()),
        "arrival_unapplied_away_gain_energy_kwh": float(-arrival_unapplied.clip(upper=0).sum()),
        "arrival_minutes_with_unapplied_away_gain_count": int(arrival_unapplied.lt(-1e-9).sum()),
        "arrival_trip_net_unapplied_energy_kwh": float(arrival_unapplied.sum()),
        "evaluation_start_soc_fraction": float(initial_soc),
        "first_minute_post_boundary_soc_fraction": float(soc.iloc[0]),
        "evaluation_end_soc_fraction": float(end_soc.iloc[-1]),
        "min_recorded_soc_fraction": float(min(initial_soc, soc.min(), end_soc.min())),
        "max_recorded_soc_fraction": float(max(initial_soc, soc.max(), end_soc.max())),
        "evaluation_start_soh_fraction": float(start_soh.iloc[0]),
        "evaluation_end_soh_fraction": float(end_soh.iloc[-1]),
        "soh_loss_fraction": float(start_soh.iloc[0] - end_soh.iloc[-1]),
        "evaluation_start_capacity_kwh": initial_capacity,
        "evaluation_end_capacity_kwh": float(end_capacity.iloc[-1]),
        "evaluation_start_chemical_energy_kwh": start_energy,
        "evaluation_end_chemical_energy_kwh": end_energy,
        "chemical_energy_change_kwh": end_energy - start_energy,
        "health_soc_min_fraction": float(health_soc_min),
        "health_soc_max_fraction": float(health_soc_max),
        "health_below_min_hours_end_state": float(end_soc.lt(health_soc_min).sum() * step_hours),
        "health_above_max_hours_end_state": float(end_soc.gt(health_soc_max).sum() * step_hours),
        "health_low_soc_fraction_hours_end_state": float(low.sum() * step_hours),
        "health_high_soc_fraction_hours_end_state": float(high.sum() * step_hours),
        "comfort_soc_min_fraction": float(comfort_soc_min),
        "departure_minutes_below_comfort_target_count": int(
            ((soc < comfort_soc_min) & departures.gt(0)).sum()),
        "departure_comfort_shortfall_soc_fraction_sum": float(departure_comfort.sum()),
        "mean_departure_comfort_shortfall_soc_fraction": (
            float(departure_comfort.sum() / departure_count) if departure_count else 0.),
        "max_departure_comfort_shortfall_soc_fraction": (
            float((comfort_soc_min - soc.loc[departures.gt(0)]).clip(lower=0).max())
            if departure_count else 0.),
        "soc_health_sampling": "end_of_minute_recorded_states",
        "service_accounting": (
            "departure shortfall is a departure-time energy-sufficiency check; "
            "arrival unapplied energy is realized trip saturation; positive unmet "
            "depletion and negative unapplied away gain are reported separately"
        ),
    }
    for suffix, key in (
        ("Battery Loss (kWh)", "battery_internal_loss_kwh"),
        ("Vehicle Standby Loss (kWh)", "vehicle_idle_standby_loss_kwh"),
        ("Fade Energy Loss (kWh)", "capacity_fade_energy_loss_kwh"),
        ("Curtailed AC Energy (kWh)", "curtailed_ac_energy_kwh"),
        ("Reference Degradation Cost ($)", "reference_degradation_cost_usd"),
        ("Battery Throughput (kWh)", "battery_throughput_kwh"),
    ):
        if f"{ev_name} {suffix}" in raw:
            result[key] = float(values(suffix).sum())
    if f"{ev_name} Electric Power (kW)" in raw:
        power = values("Electric Power (kW)")
        result.update(ev_ac_charge_energy_kwh=float(power.clip(lower=0).sum() * step_hours),
                      ev_ac_discharge_energy_kwh=float(-power.clip(upper=0).sum() * step_hours))
    return result
