"""Joint-household EV MILP path with decision-dependent V2H load following.

This builder is deliberately separate from the standalone EV controller. Its
thermal load cap is supplied by the joint optimizer for one water-by-mobility
path; the standalone EV formulation continues to use its own backed-load
forecast. The whole-home variant has no cycle, dwell, discharge-budget, or
degradation term enabled, as validated by the coordinator.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd

from examples.five_home_block.five_home_block.ev_degradation import BatteryDegradation

from .ev import (EVHorizon, _ev_comfort_cost, _ev_soc_dynamics, _ev_trip_shortfall,
                 _same_control_interval, controller_terminal_energy_value)


def build_joint_ev_path_problem(controller, timestamp, initial_soc, forecast,
                                horizon: EVHorizon, discharge_cap_kw: cp.Expression):
    """Build one MILP path using the coupled HPWH and HVAC HP/fan power cap.

    Charging and discharging retain the standalone EV's exclusive on/off
    modes. In discharge mode, AC power equals the coupled load capped at the
    inverter; SOC dynamics enforce that the battery can sustain the block.
    """
    c, o, h = controller, controller.options, horizon
    if o.formulation != "milp" or not o.v2h:
        raise ValueError("Joint EV path requires bidirectional MILP")
    if (o.enforce_cycle_limit or o.enforce_dwell_constraints
            or o.enforce_discharge_budget or o.enable_degradation):
        raise ValueError("Joint EV path requires the configured no-cycle/no-dwell health variant")
    n, E = len(h.dt), c.current_capacity
    cap = cp.reshape(discharge_cap_kw, (n,), order="C")

    x = cp.Variable(n + 1, name="x_ev")
    u_c = cp.Variable(n, nonneg=True, name="u_c")
    u_d = cp.Variable(n, nonneg=True, name="u_d")
    b_c = cp.Variable(n, boolean=True, name="b_c")
    b_d = cp.Variable(n, boolean=True, name="b_d")
    constraints = [
        x >= 0, x <= 1,
        b_c + b_d <= 1, b_c <= h.z, b_d <= h.z,
        u_c == c.ev.charger_power_kw * b_c,
        cap >= 0, cap <= o.max_discharge_kw,
        u_d <= cap,
        u_d <= o.max_discharge_kw * b_d,
        u_d >= cap - o.max_discharge_kw * (1 - b_d),
    ]
    # Mobility split boundaries do not create extra decisions inside a
    # 15-minute control interval. Ignore away blocks, as standalone EV does.
    _same_control_interval(constraints, b_c, h)
    _same_control_interval(constraints, b_d, h)

    chemical = o.charge_efficiency * u_c - u_d / o.discharge_efficiency
    ac = u_c - u_d
    shortfall, shortfall_variable, shortfall_indices, shortfall_bounds = _ev_trip_shortfall(h)
    constraints.extend(shortfall_bounds)
    served_trip = h.trip_kwh - shortfall
    dynamics, predeparture_soc = _ev_soc_dynamics(h, x, chemical, initial_soc, E, shortfall)
    constraints.extend(dynamics)

    # Preserve the standalone EV's weighted and normalized objective. The
    # old reference-backed-load term is constant with respect to all joint
    # decisions. Actual coupled discharge supplies the variable tariff credit.
    energy_cost = cp.sum(cp.multiply(h.dt, h.backed_cost
                        + cp.multiply(h.price, ac))) / h.N
    b_idle = 1 - b_c - b_d
    changes = []
    for mode, indicator in ((1, b_c), (-1, b_d), (0, b_idle)):
        previous = (cp.hstack([int(c.previous_mode == mode), indicator[:-1]])
                    if n > 1 else cp.Constant([int(c.previous_mode == mode)]))
        changes.append(cp.abs(indicator - previous))
    persistence = cp.sum(cp.multiply(h.dt, sum(changes))) / h.N

    low = cp.sum(cp.multiply(h.dt, cp.pos(o.health_soc_min - x[:-1]))) / h.N
    high = cp.sum(cp.multiply(h.dt, cp.pos(x[:-1] - o.health_soc_max))) / h.N
    comfort = (_ev_comfort_cost(h, x, o.comfort_soc_min,
                                getattr(c, "observed_peak_comfort", None))
               if o.comfort_weight else cp.Constant(0))
    terminal_value, terminal_provenance = (
        controller_terminal_energy_value(c, forecast, timestamp)
        if o.enforce_terminal_soc else (0.0, None))
    terminal = terminal_value * E * (initial_soc - x[-1]) / h.N
    terms = dict(
        electricity=o.electricity_weight * energy_cost,
        health_soc=o.health_weight * o.health_soc_weight * (low + high),
        degradation=cp.Constant(0),
        comfort=o.comfort_weight * comfort,
        trip_shortfall=o.trip_shortfall_weight * cp.sum(cp.multiply(h.dt, shortfall)) / (E * h.N),
        persistence=o.persistence_weight * persistence,
        terminal=terminal,
    )

    discharge_past = c.discharge_history(
        h.t[0] - pd.Timedelta(hours=h.history_hours), h.t[0], normalized=True
    )
    discharge_future = cp.sum(cp.multiply(h.budget_dt, u_d / o.discharge_efficiency)) / E
    battery = BatteryDegradation(c.ev.capacity_kwh, o, current_capacity_kwh=E)
    problem = cp.Problem(cp.Minimize(sum(terms.values())), constraints)
    if not problem.is_dcp() or not problem.is_mixed_integer():
        raise ValueError("Joint EV path must be a DCP-compliant MILP")
    active_first = np.flatnonzero((h.group == 0) & (h.z == 1))
    plan = dict(
        state=x, chemical=chemical, ac=ac, charge=u_c, discharge=u_d,
        fraction=b_d, charge_mode=b_c, discharge_mode=b_d,
        joint_discharge_cap_kw=cap, boundaries=h.t,
        command_index=int(active_first[0]) if len(active_first) else 0,
        N=h.N, horizon=h, starts={}, ends={}, discharge_past=discharge_past,
        measured_soh=c.current_soh, capacity_kwh=E,
        trip_shortfall_kwh=shortfall, trip_shortfall_variable=shortfall_variable,
        trip_shortfall_indices=shortfall_indices, served_trip_kwh=served_trip,
        terminal_value_usd_per_kwh=terminal_value, terminal_value_provenance=terminal_provenance,
        initial_soc=initial_soc,
        rainflow_theta=None, battery=battery, health_low=low, health_high=high,
        discharge_future=discharge_future, objective_terms=terms, **terms,
    )
    return problem, plan
