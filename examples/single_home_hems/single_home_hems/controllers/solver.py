"""Quiet, bounded OCP solves shared by EV and HPWH (not model training).

Native backend limits exclude CVXPY construction/canonicalization and may have
small backend overhead. They are not a process-level wall-clock watchdog.
"""
from copy import deepcopy
import time
import cvxpy as cp
import numpy as np

LIMIT_SECONDS = 300.0


def solve_ocp(problem, solver, options=None, *, time_limit_seconds=LIMIT_SECONDS):
    """One backend call; EV outer iterations may request the remaining budget."""
    limit = min(LIMIT_SECONDS, float(time_limit_seconds))
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("Solver time budget must be finite and positive")
    options = deepcopy(dict(options or {}))
    options.pop("solver", None)
    options.update(verbose=False)
    options.pop("solver_verbose", None)
    if solver == "SCIPY":
        scipy = options.setdefault("scipy_options", {})
        scipy.update(time_limit=limit, disp=False)
    elif solver == "CLARABEL":
        options["time_limit"] = limit
    elif solver == "GUROBI":
        options.update(TimeLimit=limit, OutputFlag=0, LogToConsole=0, LogFile="")
    else:
        raise ValueError(f"Unsupported OCP backend {solver}; use SCIPY, CLARABEL or GUROBI")
    started = time.perf_counter()
    diagnostic = dict(solver=solver, limit_seconds=limit, timed_out=False,
                      limit_reached=False, fallback=True, accepted_incumbent=False)
    problem.hems_diagnostic = diagnostic
    try:
        problem.solve(solver=solver, **options)
    finally:
        diagnostic["solve_call_seconds"] = time.perf_counter() - started
        stats = getattr(problem, "solver_stats", None)
        backend_time = getattr(stats, "solve_time", None)
        diagnostic["backend_solve_seconds"] = backend_time
        extra = getattr(stats, "extra_stats", None)
        # USER_LIMIT includes iteration limits. Do not label every limit a timeout.
        diagnostic["limit_reached"] = problem.status == cp.USER_LIMIT
        diagnostic["timed_out"] = bool(
            getattr(extra, "Status", None) == 9 or
            (diagnostic["limit_reached"] and
             (backend_time or diagnostic["solve_call_seconds"]) >= limit - .5))


def usable_solution(problem, tolerance=1e-4):
    """Accept finite feasible incumbents, never silently label them optimal."""
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE, cp.USER_LIMIT}:
        return False
    try:
        valid = all(v.value is not None and np.isfinite(v.value).all() for v in problem.variables())
        valid = valid and all(np.max(np.abs(c.violation())) <= tolerance for c in problem.constraints)
        for v in problem.variables():
            if v.attributes.get("boolean") or v.attributes.get("integer"):
                valid = valid and np.max(np.abs(v.value - np.round(v.value))) <= tolerance
    except (ValueError, TypeError):
        valid = False
    diagnostic = getattr(problem, "hems_diagnostic", {})
    diagnostic.update(fallback=not bool(valid), accepted_incumbent=bool(valid and problem.status == cp.USER_LIMIT))
    return bool(valid)
