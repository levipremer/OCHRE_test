"""Quiet EV OCP solves with bounded defaults and an explicit uncapped opt-in.

Native backend limits exclude CVXPY construction/canonicalization and may have
small backend overhead. They are not a process-level wall-clock watchdog.
"""
from copy import deepcopy
import time
import cvxpy as cp
import numpy as np

LIMIT_SECONDS = 300.0


def solve_ocp(problem, solver, options=None, *, time_limit_seconds=LIMIT_SECONDS,
              allow_unlimited_time=False, solve_callback=None):
    """One backend call; an uncapped solve requires an explicit opt-in and None."""
    if type(allow_unlimited_time) is not bool:
        raise ValueError("allow_unlimited_time must be a boolean")
    if time_limit_seconds is None:
        if not allow_unlimited_time:
            raise ValueError("An unlimited solver time budget requires explicit opt-in")
        limit = None
    else:
        limit = float(time_limit_seconds)
        if not np.isfinite(limit) or limit <= 0:
            raise ValueError("Solver time budget must be finite and positive")
        if allow_unlimited_time:
            raise ValueError("An unlimited solver time budget must be None")
    options = deepcopy(dict(options or {}))
    # EVOptions normally resolves legacy backend limits into its single budget.
    # Standalone callers must likewise never have an explicit shorter limit raised.
    requested = (options.get("scipy_options", {}).get("time_limit") if solver == "SCIPY"
                 else options.get("TimeLimit" if solver == "GUROBI" else "time_limit"))
    if requested is not None:
        requested = float(requested)
        if not np.isfinite(requested) or requested <= 0:
            raise ValueError("Backend time limit must be finite and positive")
        if limit is None:
            raise ValueError("An unlimited solve cannot have a backend time limit")
        limit = min(limit, requested)
    options.pop("solver", None)
    options.update(verbose=False)
    options.pop("solver_verbose", None)
    if solver == "SCIPY":
        scipy = options.setdefault("scipy_options", {})
        scipy.update(disp=False)
        if limit is not None:
            scipy["time_limit"] = limit
    elif solver == "CLARABEL":
        if limit is not None:
            options["time_limit"] = limit
    elif solver == "GUROBI":
        options.update(OutputFlag=0, LogToConsole=0, LogFile="")
        if limit is not None:
            options["TimeLimit"] = limit
    else:
        raise ValueError(f"Unsupported OCP backend {solver}; use SCIPY, CLARABEL or GUROBI")
    started = time.perf_counter()
    diagnostic = dict(solver=solver, limit_seconds=limit, backend_limit_seconds=limit,
                      unlimited_time=limit is None, timed_out=False,
                      limit_reached=False, fallback=True, accepted_incumbent=False)
    problem.hems_diagnostic = diagnostic
    try:
        if solve_callback is None:
            problem.solve(solver=solver, **options)
        else:
            # Alternative compilers receive the same prepared backend options
            # and retain this shared time-budget and diagnostic policy.
            solve_callback(problem, solver=solver, **options)
    finally:
        diagnostic["solve_call_seconds"] = time.perf_counter() - started
        stats = getattr(problem, "solver_stats", None)
        backend_time = getattr(stats, "solve_time", None)
        diagnostic["backend_solve_seconds"] = backend_time
        extra = getattr(stats, "extra_stats", None)
        # USER_LIMIT includes iteration limits. Do not label every limit a timeout.
        diagnostic["limit_reached"] = diagnostic["limit_reached"] or problem.status == cp.USER_LIMIT
        diagnostic["timed_out"] = diagnostic["timed_out"] or bool(
            getattr(extra, "Status", None) == 9 or
            (limit is not None and diagnostic["limit_reached"] and
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
