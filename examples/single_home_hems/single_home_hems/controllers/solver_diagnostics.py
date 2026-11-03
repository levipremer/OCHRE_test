"""Read rejected solver points without repairing or accepting their primals.

The returned record is JSON safe. Nonfinite raw numbers use explicit string
markers (``NaN``, ``+Infinity``, ``-Infinity``), accompanied by finiteness flags.
No solver, variable assignment, projection, or constraint mutation occurs here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


_NATIVE_ATTRIBUTES = (
    "Status", "SolCount", "ObjVal", "ObjBound", "MIPGap", "Runtime",
    "MaxVio", "BoundVio", "BoundSVio", "BoundVioIndex", "BoundSVioIndex",
    "ConstrVio", "ConstrSVio", "ConstrVioIndex", "ConstrSVioIndex",
    "ConstrResidual", "ConstrSResidual", "ConstrResidualIndex",
    "ConstrSResidualIndex", "DualVio", "DualSVio", "DualVioIndex",
    "DualSVioIndex", "DualResidual", "DualSResidual", "ComplVio",
    "IntVio", "IntVioIndex",
)


def _error(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


def _json_value(value: Any) -> Any:
    """Copy numeric values, retaining nonfinite identity in ordinary JSON."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float):
        if np.isnan(value):
            return "NaN"
        if np.isposinf(value):
            return "+Infinity"
        if np.isneginf(value):
            return "-Infinity"
        return value
    if isinstance(value, complex):
        return {"real": _json_value(value.real), "imag": _json_value(value.imag)}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if hasattr(value, "toarray"):
        return _json_value(value.toarray())
    return str(value)


def _array_summary(value: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "raw_value": _json_value(value), "missing": value is None,
        "finite": False, "nonfinite_count": 0, "nonfinite_locations": [],
    }
    if value is None:
        return result
    array = np.asarray(value.toarray() if hasattr(value, "toarray") else value)
    finite = np.isfinite(array)
    result.update(
        finite=bool(np.all(finite)),
        nonfinite_count=int(np.count_nonzero(~finite)),
        nonfinite_locations=np.argwhere(~finite).tolist(),
    )
    return result


def _maximum(value: Any) -> tuple[float | None, list[int] | None]:
    array = np.asarray(value)
    if array.size == 0:
        return 0.0, None
    # Nonfinite locations are separately retained; the finite maximum still
    # identifies useful residuals when only part of a point is nonfinite.
    absolute = np.abs(array)
    finite = np.isfinite(absolute)
    if not np.any(finite):
        return None, None
    flat_index = int(np.argmax(np.where(finite, absolute, -np.inf)))
    return float(absolute.flat[flat_index]), [
        int(coordinate) for coordinate in np.unravel_index(flat_index, array.shape)
    ]


def _integrality(value: Any, attributes: Mapping[str, Any]) -> dict[str, Any]:
    array = np.asarray(value)
    mask = np.zeros(array.shape, dtype=bool)
    kinds = []
    for kind in ("boolean", "integer"):
        indices = attributes.get(kind, False)
        if isinstance(indices, (bool, np.bool_)):
            if indices:
                mask[...] = True
                kinds.append(kind)
        elif indices is not None and len(indices):
            kinds.append(kind)
            # CVXPY keeps indexed integrality as a list of coordinate tuples.
            for index in indices:
                mask[tuple(index)] = True
    locations = np.argwhere(mask)
    selected = array[mask]
    residuals = np.abs(selected - np.round(selected))
    maximum, index = _maximum(residuals)
    worst = None if index is None else locations[index[0]].tolist()
    return {
        "kinds": kinds, "element_count": int(selected.size),
        "max_residual": maximum, "worst_location": worst,
        "nonfinite_count": int(np.count_nonzero(~np.isfinite(selected))),
    }


def _native_diagnostics(problem: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "available": False, "has_incumbent": None, "attributes": {},
        "attribute_errors": {}, "variables": [], "errors": [],
    }
    try:
        stats = problem.solver_stats
        model = getattr(stats, "extra_stats", None)
        if isinstance(model, Mapping):
            model = next((model[key] for key in ("model", "gurobi_model", "native_model")
                          if key in model), None)
        if model is None or not callable(getattr(model, "getVars", None)):
            return result
        result["available"] = True
        result["model_type"] = type(model).__name__
        for name in _NATIVE_ATTRIBUTES:
            try:
                result["attributes"][name] = _json_value(getattr(model, name))
            except Exception as error:
                result["attribute_errors"][name] = _error(error)
        solution_count = result["attributes"].get("SolCount")
        if isinstance(solution_count, (int, float)):
            result["has_incumbent"] = solution_count > 0
        if result["has_incumbent"]:
            for index, variable in enumerate(model.getVars()):
                item = {"index": index, "errors": {}}
                for name in ("VarName", "X", "Obj", "LB", "UB", "VType"):
                    try:
                        item[name] = _json_value(getattr(variable, name))
                    except Exception as error:
                        item["errors"][name] = _error(error)
                result["variables"].append(item)
    except Exception as error:
        result["errors"].append(_error(error))
    return result


def _collect_primal_diagnostics(problem: Any) -> dict[str, Any]:
    """Return an observational, JSON-safe snapshot of a candidate primal point.

    Failures to inspect an objective, variable, constraint, or native quality
    attribute are recorded as data. This helper does not call a solver, assign
    values, clip arrays, or decide whether a candidate may be dispatched.
    """
    result: dict[str, Any] = {
        "schema_version": 1, "observational": True, "status": None,
        "objective": {}, "variables": [], "constraints": [],
        "max_integrality_residual": None, "max_constraint_violation": None,
        "errors": [],
    }
    try:
        result["status"] = _json_value(problem.status)
    except Exception as error:
        result["errors"].append(f"status: {_error(error)}")
    try:
        result["objective"] = _array_summary(problem.objective.value)
    except Exception as error:
        result["objective"] = {"error": _error(error), "finite": False}

    try:
        variables = problem.variables()
    except Exception as error:
        result["errors"].append(f"variables: {_error(error)}")
        variables = []
    for index, variable in enumerate(variables):
        item: dict[str, Any] = {"index": index, "errors": {}}
        for name, accessor in (
            ("id", lambda: variable.id), ("name", lambda: variable.name()),
            ("shape", lambda: variable.shape),
            ("attributes", lambda: variable.attributes),
        ):
            try:
                item[name] = _json_value(accessor())
            except Exception as error:
                item["errors"][name] = _error(error)
        try:
            value = variable.value
            item.update(_array_summary(value))
            if value is not None:
                item["integrality"] = _integrality(value, variable.attributes)
                maximum = item["integrality"]["max_residual"]
                current = result["max_integrality_residual"]
                if maximum is not None and (current is None or maximum > current):
                    result["max_integrality_residual"] = maximum
                    result["worst_integrality_variable_id"] = item.get("id")
                    result["worst_integrality_location"] = item["integrality"]["worst_location"]
        except Exception as error:
            item["errors"]["value_or_integrality"] = _error(error)
        result["variables"].append(item)

    try:
        constraints = problem.constraints
    except Exception as error:
        result["errors"].append(f"constraints: {_error(error)}")
        constraints = []
    for index, constraint in enumerate(constraints):
        item = {"index": index, "type": type(constraint).__name__, "errors": {}}
        for name in ("id", "shape"):
            try:
                item[name] = _json_value(getattr(constraint, name))
            except Exception as error:
                item["errors"][name] = _error(error)
        try:
            item["missing"] = any(argument.value is None for argument in constraint.args)
        except Exception as error:
            item["errors"]["missing_values"] = _error(error)
        try:
            violation = constraint.violation()
            item.update(_array_summary(violation))
            item["raw_violation"] = item["raw_value"]
            maximum, worst = _maximum(violation) if violation is not None else (None, None)
            item.update(violation_max=maximum, worst_location=worst)
            if violation is not None:
                absolute = np.abs(np.asarray(violation))
                raw_maximum = np.max(absolute) if absolute.size else 0.0
                item["raw_violation_max"] = _json_value(raw_maximum)
                raw_index = int(np.argmax(absolute)) if absolute.size else None
                item["raw_violation_worst_location"] = (
                    None if raw_index is None else [
                        int(coordinate) for coordinate in np.unravel_index(raw_index, absolute.shape)
                    ]
                )
            current = result["max_constraint_violation"]
            if maximum is not None and (current is None or maximum > current):
                result["max_constraint_violation"] = maximum
                result["worst_constraint_index"] = index
                result["worst_constraint_location"] = worst
        except Exception as error:
            item["errors"]["violation"] = _error(error)
        result["constraints"].append(item)

    result["native_solver"] = _native_diagnostics(problem)
    return result


def collect_primal_diagnostics(problem: Any) -> dict[str, Any]:
    """Read a candidate point; inspection errors never decide dispatch.

    ``violation_max`` and ``max_constraint_violation`` summarize finite
    residuals. Each constraint also preserves ``raw_violation_max`` and its
    location, including explicit NaN/Infinity markers, so nonfinite entries
    cannot silently appear feasible. Raw values are copied without rounding.
    """
    try:
        return _collect_primal_diagnostics(problem)
    except Exception as error:
        # Defensive boundary for unusual solver wrappers and malformed points.
        # Diagnostic collection must never disrupt the caller's fallback path.
        return {
            "schema_version": 1, "observational": True,
            "collection_failed": True, "errors": [_error(error)],
        }
