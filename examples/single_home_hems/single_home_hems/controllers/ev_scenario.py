"""Two-stage EV MPC: shared first command, path-specific future recourse.

min max_s J_comfort,s + max_s J_trip,s + mean_s(all remaining costs).
Dynamics, optional constraints, and cost normalization come from ev.py.
"""
import time
import cvxpy as cp
import numpy as np

from .solver import LIMIT_SECONDS, solve_ocp, usable_solution


def build_scenario_problem(controller, timestamp, initial_soc, scenarios):
    """Separate worst-path comfort/trip costs; arithmetic mean of other costs."""
    from .ev import build_ev_problem
    if not scenarios.forecasts:
        raise ValueError("EV scenarios need paired Forecast objects")
    constraints, plans = [], []
    for forecast in scenarios.forecasts:
        problem, plan = build_ev_problem(controller, timestamp, initial_soc, forecast)
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
        terms[key] = cp.max(cp.hstack(costs)) if key in {"comfort", "trip_shortfall"} else sum(costs)/len(plans)
    problem = cp.Problem(cp.Minimize(sum(terms.values())), constraints)
    if not problem.is_dcp():
        raise ValueError("EV scenario formulation is not DCP")
    # Keep the action/reporting contract; detailed paths are stored separately.
    combined = dict(plans[0], objective_terms=terms, scenario_plans=plans,
                    scenario_metadata=scenarios.diagnostics,
                    scenario_ids=scenarios.candidate_ids or tuple(str(i) for i in range(len(plans))))
    combined.update(terms)
    return problem, combined


def solve_rainflow(controller, problem, plan, options, deadline):
    """Multicut outer approximation with independent per-path aging certification.

    Feasible exact objectives bound the joint problem above; the joint master
    bound is below. One 300-s budget covers every path and master iteration.
    Never update observed history with a forecast scenario.
    """
    from .ev import master_lower_bound
    o = controller.options
    paths = plan["scenario_plans"]
    history = list(controller.soc_history)
    initial = float(plan["initial_soc"])
    if not history or abs(history[-1]-initial) > 1e-10:
        history.append(initial)
    prefix = np.asarray(history[:-1])
    historical = [p["battery"].rainflow_cost_gradient(history)[0] for p in paths]
    scales = [o.health_weight/p["N"]/len(paths) for p in paths]
    base = sum(v for k, v in plan["objective_terms"].items() if k != "degradation")
    constraints = list(problem.constraints)
    planes = [[] for _ in paths]
    lower, upper = -np.inf, np.inf
    best, best_values, best_soc, best_cost = None, None, None, None
    iterations, cuts, elapsed, backend = 0, 0, 0.0, 0.0
    backend_known, converged, timed_out = False, False, False
    absolute = o.rainflow_absolute_tolerance
    tolerance = o.mip_relative_gap if o.formulation == "milp" else o.rainflow_relative_tolerance
    errors = np.full(len(paths), np.inf)
    limits = np.full(len(paths), np.nan)
    last_error = None

    def evaluate(socs):
        costs, gradients = [], []
        for p, past, soc in zip(paths, historical, socs):
            total, gradient = p["battery"].rainflow_cost_gradient(np.r_[prefix, soc])
            if total-past < -absolute:
                raise RuntimeError("Rainflow incremental cost became negative")
            costs.append(max(0., total-past))
            gradients.append(gradient[len(prefix):])
        return np.asarray(costs), gradients

    def accuracy(socs, costs):
        envelope = [max([0., *[value+g@(soc-anchor) for value, g, anchor in cuts_s]])
                    for soc, cuts_s in zip(socs, planes)]
        return np.maximum(0., costs-envelope), absolute+o.rainflow_relative_tolerance*np.abs(costs)

    for _ in range(o.rainflow_max_iterations):
        remaining = deadline-time.perf_counter()
        if remaining <= 0:
            timed_out = True
            break
        master = cp.Problem(cp.Minimize(base+sum(a*p["rainflow_theta"] for a, p in zip(scales, paths))), constraints)
        iterations += 1
        try:
            solve_ocp(master, o.solver, options, time_limit_seconds=remaining)
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
        socs = [p["rainflow_state"].value.copy() for p in paths]
        costs, gradients = evaluate(socs)
        objective = float(base.value)+float(np.dot(scales, costs))
        if objective < upper:
            upper, best = objective, master
            best_soc, best_cost = [s.copy() for s in socs], costs.copy()
            best_values = {v.id: np.array(v.value, copy=True) for v in master.variables()}
            for p, cost in zip(paths, costs):
                best_values[p["rainflow_theta"].id] = np.asarray(cost)
        if np.isfinite(lower) and lower > upper+absolute+1e-7*max(1, abs(upper)):
            raise RuntimeError("Rainflow lower bound exceeds exact feasible objective")
        errors, limits = accuracy(best_soc, best_cost)
        certified = max(0., upper-lower) <= absolute+tolerance*max(abs(upper), absolute)
        if certified and np.all(errors <= limits):
            converged = True
            break
        if certified:
            socs, costs = best_soc, best_cost
            _, gradients = evaluate(socs)
        for p, value, gradient, soc, cuts_s in zip(paths, costs, gradients, socs, planes):
            constraints.append(p["rainflow_theta"] >= value+gradient@(p["rainflow_state"]-soc))
            cuts_s.append((value, gradient.copy(), soc.copy()))
            cuts += 1
    if best is not None:
        problem = best
        for variable in best.variables():
            variable.save_value(best_values[variable.id])
        errors, limits = accuracy(best_soc, best_cost)
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
        limit_seconds=LIMIT_SECONDS, fallback=not converged,
        accepted_incumbent=bool(converged and problem.status == cp.USER_LIMIT),
        outer_status="rainflow_converged" if converged else (last_error or "rainflow_not_converged"))
    problem.hems_diagnostic = info
    return problem, converged
