"""Two-stage EV MPC: shared first command, path-specific future recourse.

min max_s J_comfort,s + mean_s(sum of all remaining weighted costs).
Standalone scenario paths use ev.py; whole-home paths use ev_joint.py with
the same EV state and cost normalization and a coupled thermal discharge cap.
"""
import time
import cvxpy as cp
import numpy as np
import pandas as pd

from .solver import solve_ocp, usable_solution


def build_scenario_problem(controller, timestamp, initial_soc, scenarios):
    """Share the first command; minimize worst-path comfort and mean other costs."""
    from .ev import build_ev_problem
    if not scenarios.forecasts:
        raise ValueError("EV scenarios need paired Forecast objects")
    constraints, plans = [], []
    for forecast in scenarios.forecasts:
        problem, plan = build_ev_problem(controller, timestamp, initial_soc, forecast,
                                         trip_timing="departure")
        constraints.extend(problem.constraints)
        plans.append(plan)

    # Share the first CONTROL interval, including its mobility-split blocks.
    # Away blocks remain physically zero; command sharing concerns active blocks.
    first = None
    for plan in plans:
        h = plan["horizon"]
        active = np.flatnonzero((h.group == 0) & (h.z == 1))
        if not len(active):
            continue
        j = int(active[0])
        command = ([plan["charge_mode"][j], plan["discharge_mode"][j]]
                   if controller.options.formulation == "milp" else [plan["chemical"][j]])
        if first is not None:
            constraints.extend(a == b for a, b in zip(command, first))
        else:
            first = command

    terms = {}
    for key in plans[0]["objective_terms"]:
        costs = [p["objective_terms"][key] for p in plans]
        terms[key] = cp.max(cp.hstack(costs)) if key == "comfort" else sum(costs)/len(plans)
    problem = cp.Problem(cp.Minimize(sum(terms.values())), constraints)
    if not problem.is_dcp():
        raise ValueError("EV scenario formulation is not DCP")
    # Keep the action/reporting contract; detailed paths are stored separately.
    combined = dict(plans[0], objective_terms=terms, scenario_plans=plans,
                    shared_first_command=first,
                    scenario_metadata={**scenarios.diagnostics,
                                       "trip_energy_timing": "departure"},
                    scenario_ids=scenarios.candidate_ids or tuple(str(i) for i in range(len(plans))))
    combined.update(terms)
    return problem, combined


def build_joint_ev_scenario_problem(
    controller, timestamp, initial_soc, scenarios, water_power_kw,
    hvac_hp_fan_power_kw, thermal_step, *, eligible_power_upper_kw,
    hub_other_kw=None, perfect_forecasts=False,
):
    """Build paired EV recourse paths against configured HPWH load scenarios.

    Eligible load is HPWH electricity, HVAC compressor/fan and Hub-other.
    HVAC resistance and Outside-Hub are absent. The same thermal choices
    appear in all mobility paths for a given water scenario.
    A shared binary command covers the first active EV control interval.
    """
    from .ev import EVHorizon
    from .ev_joint import build_joint_ev_path_problem

    if controller.options.formulation != "milp" or not controller.options.v2h:
        raise ValueError("Joint load-following EV scenarios require bidirectional MILP")
    if controller.options.enforce_dwell_constraints or controller.options.enforce_cycle_limit:
        raise ValueError("Joint EV builder needs the configured health-cost variant without dwell/cycle limits")
    ev_count = len(scenarios.forecasts)
    if ev_count not in ((1,) if perfect_forecasts else (5, 7)):
        raise ValueError("Joint EV builder requires five or seven EV forecasts")
    water = cp.Constant(water_power_kw) if isinstance(water_power_kw, np.ndarray) else water_power_kw
    hvac = cp.Constant(hvac_hp_fan_power_kw) if isinstance(hvac_hp_fan_power_kw, np.ndarray) else hvac_hp_fan_power_kw
    allowed_water_counts = (1,) if perfect_forecasts else (5, 7, 10)
    if (len(water.shape) != 2 or water.shape[0] not in allowed_water_counts
            or len(hvac.shape) != 1 or hvac.shape[0] != water.shape[1]):
        raise ValueError("Joint EV eligible load needs configured scenarios-by-steps water and steps HVAC power")
    water_count = water.shape[0]
    steps = water.shape[1]
    hub = np.zeros(steps) if hub_other_kw is None else np.asarray(hub_other_kw, dtype=float).reshape(-1)
    if hub.shape != (steps,) or not np.isfinite(hub).all() or (hub < 0).any():
        raise ValueError("Joint EV Hub-other forecast needs nonnegative 15-minute power")
    thermal_step = pd.Timedelta(thermal_step)
    timestamp = pd.Timestamp(timestamp)
    if thermal_step <= pd.Timedelta(0) or steps * thermal_step != controller.horizon:
        raise ValueError("Joint EV and thermal horizons must use the same grid")
    upper = float(eligible_power_upper_kw)
    inverter = float(controller.options.max_discharge_kw)
    if not np.isfinite(upper) or upper <= 0 or not np.isfinite(inverter) or inverter <= 0:
        raise ValueError("Eligible-load and inverter limits must be positive and finite")

    # The physical model power limits give a tight bound for the min() MILP.
    # The cap is shared by EV mobility paths with the same water scenario.
    eligible = water + cp.vstack([hvac + hub for _ in range(water_count)])
    constraints = [eligible >= 0, eligible <= upper]
    if upper <= inverter:
        capped = eligible
    else:
        capped = cp.Variable((water_count, steps), nonneg=True, name="joint_ev_eligible_cap_kw")
        clipped = cp.Variable((water_count, steps), boolean=True, name="joint_ev_inverter_clipped")
        constraints += [
            capped <= eligible, capped <= inverter,
            capped >= eligible - upper * clipped,
            capped >= inverter - inverter * (1 - clipped),
        ]

    boundaries = pd.date_range(timestamp, periods=steps + 1, freq=thermal_step)
    horizons = [EVHorizon.from_forecast(controller, timestamp, forecast,
                                        trip_timing="departure")
                for forecast in scenarios.forecasts]
    block_indices = []
    for h in horizons:
        if h.t[0] != boundaries[0] or h.t[-1] != boundaries[-1]:
            raise ValueError("EV mobility path does not span the complete thermal horizon")
        indices = np.searchsorted(boundaries.asi8, h.t[:-1].asi8, side="right") - 1
        if np.any(indices < 0) or np.any(indices >= steps) or np.any(h.t[1:].asi8 > boundaries.asi8[indices + 1]):
            raise ValueError("EV mobility block crosses a thermal power interval")
        block_indices.append(indices)

    plans, identifiers = [], []
    ev_ids = scenarios.candidate_ids or tuple(str(i) for i in range(ev_count))
    for water_index in range(water_count):
        for ev_index, (forecast, h, indices) in enumerate(zip(scenarios.forecasts, horizons, block_indices)):
            cap_by_block = cp.hstack([capped[water_index, int(i)] for i in indices])
            subproblem, plan = build_joint_ev_path_problem(
                controller, timestamp, initial_soc, forecast, h, cap_by_block,
            )
            constraints.extend(subproblem.constraints)
            plan.update(water_scenario_index=water_index, ev_scenario_index=ev_index)
            plans.append(plan)
            identifiers.append(f"water_{water_index + 1}|ev_{ev_ids[ev_index]}")

    # Share only the currently implementable charge/discharge mode. Each
    # water-by-EV path keeps its own later mode and battery trajectory.
    first = None
    for plan in plans:
        h = plan["horizon"]
        active = np.flatnonzero((h.group == 0) & (h.z == 1))
        if not len(active):
            continue
        j = int(active[0])
        command = [plan["charge_mode"][j], plan["discharge_mode"][j]]
        if first is None:
            first = command
        else:
            constraints.extend(a == b for a, b in zip(command, first))

    terms = {}
    for key in plans[0]["objective_terms"]:
        costs = [p["objective_terms"][key] for p in plans]
        terms[key] = cp.max(cp.hstack(costs)) if key == "comfort" else sum(costs) / len(plans)
    problem = cp.Problem(cp.Minimize(sum(terms.values())), constraints)
    if not problem.is_dcp() or not problem.is_mixed_integer():
        raise ValueError("Joint EV scenario formulation must be a DCP-compliant MILP")
    combined = dict(plans[0], objective_terms=terms, scenario_plans=plans,
                    shared_first_command=first, scenario_metadata={**scenarios.diagnostics,
                        "trip_energy_timing": "departure",
                        "water_scenario_count": water_count, "ev_scenario_count": ev_count,
                        "joint_scenario_count": water_count * ev_count,
                        "eligible_load_source": "HPWH + HVAC compressor/fan + Hub-other forecast"},
                    scenario_ids=tuple(identifiers), scenario_shape=(water_count, ev_count),
                    eligible_load_kw=eligible, eligible_cap_kw=capped)
    combined.update(terms)
    return problem, combined


def solve_rainflow(controller, problem, plan, options, deadline):
    """Multicut outer approximation with independent per-path aging certification.

    Feasible exact objectives bound the joint problem above; the joint master
    bound is below. One configured budget covers every path and master iteration.
    Never update observed history with a forecast scenario.
    """
    from .ev import _rainflow_path_cost_gradient, master_lower_bound
    o = controller.options
    paths = plan["scenario_plans"]
    history = list(controller.soc_history)
    initial = float(plan["initial_soc"])
    if not history or abs(history[-1]-initial) > 1e-10:
        history.append(initial)
    historical = [p["battery"].rainflow_cost_gradient(history)[0] for p in paths]
    scales = [o.health_weight/p["N"]/len(paths) for p in paths]
    base = sum(v for k, v in plan["objective_terms"].items() if k != "degradation")
    constraints = list(problem.constraints)
    planes = [[] for _ in paths]
    shortfall_expressions = [p.get("trip_shortfall_kwh",
                                  cp.Constant(np.zeros(len(p["horizon"].dt))))
                             for p in paths]
    lower, upper = -np.inf, np.inf
    best, best_values, best_soc, best_cost = None, None, None, None
    best_shortfall = None
    iterations, cuts, elapsed, backend = 0, 0, 0.0, 0.0
    backend_known, converged, timed_out = False, False, False
    absolute = o.rainflow_absolute_tolerance
    tolerance = o.mip_relative_gap if o.formulation == "milp" else o.rainflow_relative_tolerance
    errors = np.full(len(paths), np.inf)
    limits = np.full(len(paths), np.nan)
    last_error = None

    def evaluate(socs, shortfalls):
        costs, gradients, shortfall_gradients = [], [], []
        for p, past, soc, eps in zip(paths, historical, socs, shortfalls):
            total, gradient, shortfall_gradient = _rainflow_path_cost_gradient(
                p["battery"], history, p["horizon"], soc, p["capacity_kwh"], eps)
            if total-past < -absolute:
                raise RuntimeError("Rainflow incremental cost became negative")
            costs.append(max(0., total-past))
            gradients.append(gradient)
            shortfall_gradients.append(shortfall_gradient)
        return np.asarray(costs), gradients, shortfall_gradients

    def accuracy(socs, shortfalls, costs):
        envelope = [max([0., *[value+g@(soc-anchor)+eg@(eps-eps_anchor)
                              for value, g, anchor, eg, eps_anchor in cuts_s]])
                    for soc, eps, cuts_s in zip(socs, shortfalls, planes)]
        return np.maximum(0., costs-envelope), absolute+o.rainflow_relative_tolerance*np.abs(costs)

    for _ in range(o.rainflow_max_iterations):
        remaining = deadline-time.perf_counter()
        if remaining <= 0:
            timed_out = True
            break
        master = cp.Problem(cp.Minimize(base+sum(a*p["rainflow_theta"] for a, p in zip(scales, paths))), constraints)
        iterations += 1
        try:
            solve_ocp(master, o.solver, options,
                      time_limit_seconds=None if o.allow_unlimited_solver_time else remaining,
                      allow_unlimited_time=o.allow_unlimited_solver_time)
        except cp.error.SolverError as error:
            last_error = str(error)
        diagnostic = getattr(master, "hems_diagnostic", {})
        elapsed += diagnostic.get("solve_call_seconds", 0)
        if diagnostic.get("backend_solve_seconds") is not None:
            backend += diagnostic["backend_solve_seconds"]
            backend_known = True
        timed_out |= diagnostic.get("timed_out", False)
        problem = master
        if last_error or not usable_solution(master):
            break
        lower = max(lower, master_lower_bound(master, o.solver))
        socs = [p["state"].value.copy() for p in paths]
        shortfalls = [np.array(eps.value, copy=True) for eps in shortfall_expressions]
        costs, gradients, shortfall_gradients = evaluate(socs, shortfalls)
        objective = float(base.value)+float(np.dot(scales, costs))
        if objective < upper:
            upper, best = objective, master
            best_soc, best_cost = [s.copy() for s in socs], costs.copy()
            best_shortfall = [eps.copy() for eps in shortfalls]
            best_values = {v.id: np.array(v.value, copy=True) for v in master.variables()}
            for p, cost in zip(paths, costs):
                best_values[p["rainflow_theta"].id] = np.asarray(cost)
        if np.isfinite(lower) and lower > upper+absolute+1e-7*max(1, abs(upper)):
            raise RuntimeError("Rainflow lower bound exceeds exact feasible objective")
        errors, limits = accuracy(best_soc, best_shortfall, best_cost)
        certified = max(0., upper-lower) <= absolute+tolerance*max(abs(upper), absolute)
        if certified and np.all(errors <= limits):
            converged = True
            break
        if certified:
            socs, shortfalls, costs = best_soc, best_shortfall, best_cost
            _, gradients, shortfall_gradients = evaluate(socs, shortfalls)
        for p, value, gradient, soc, eps_expression, eps, eps_gradient, cuts_s in zip(
                paths, costs, gradients, socs, shortfall_expressions, shortfalls,
                shortfall_gradients, planes):
            constraints.append(p["rainflow_theta"] >= value+gradient@(p["state"]-soc)
                               + eps_gradient@(eps_expression-eps))
            cuts_s.append((value, gradient.copy(), soc.copy(), eps_gradient.copy(), eps.copy()))
            cuts += 1
    if best is not None:
        problem = best
        for variable in best.variables():
            variable.save_value(best_values[variable.id])
        errors, limits = accuracy(best_soc, best_shortfall, best_cost)
    finite = np.isfinite(upper) and np.isfinite(lower)
    info = dict(getattr(problem, "hems_diagnostic", {}))
    info.update(rainflow_iterations=iterations, rainflow_cuts=cuts,
        rainflow_upper_bound=upper, rainflow_lower_bound=lower,
        rainflow_relative_gap=max(0., upper-lower)/max(abs(upper), absolute, 1e-12) if finite else np.inf,
        rainflow_absolute_gap=max(0., upper-lower) if finite else np.inf,
        rainflow_converged=converged, rainflow_model_certified=bool(np.all(errors <= limits)),
        rainflow_approximation_error_usd=float(np.max(errors)),
        rainflow_scenario_errors_usd=errors.tolist(), rainflow_scenario_tolerances_usd=limits.tolist(),
        historical_rainflow_cost_usd=float(np.mean(historical)), exact_objective=upper,
        solve_call_seconds=elapsed, backend_solve_seconds=backend if backend_known else None,
        timed_out=bool(timed_out or time.perf_counter() >= deadline), limit_reached=not converged,
        limit_seconds=o.solver_time_limit_seconds, fallback=not converged,
        accepted_incumbent=bool(converged and problem.status == cp.USER_LIMIT),
        outer_status="rainflow_converged" if converged else (last_error or "rainflow_not_converged"))
    problem.hems_diagnostic = info
    return problem, converged
