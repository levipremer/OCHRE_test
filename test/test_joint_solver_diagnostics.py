"""Rejected primal evidence is preserved without solving or changing a point."""

from copy import deepcopy
import json
from types import SimpleNamespace

import cvxpy as cp
import numpy as np

from examples.single_home_hems.single_home_hems.controllers.solver_diagnostics import (
    collect_primal_diagnostics,
)


def test_infeasible_raw_primal_records_constraint_and_integrality_evidence():
    binary = cp.Variable(2, boolean=True, name="device_on")
    power = cp.Variable(2, nonneg=True, name="power_kw")
    constraints = [power <= 2 * binary, cp.sum(power) == 1]
    problem = cp.Problem(cp.Minimize(cp.sum(power)), constraints)
    # Solvers can supply points outside attribute tolerances. Save an artificial
    # raw point directly, without asking CVXPY to project or validate it.
    binary.save_value(np.array([0.25, 1.00002]))
    power.save_value(np.array([2.0, -0.25]))
    before_values = [value.value.copy() for value in problem.variables()]
    before_attributes = [deepcopy(value.attributes) for value in problem.variables()]

    diagnostics = collect_primal_diagnostics(problem)

    assert diagnostics["objective"]["raw_value"] == 1.75
    assert diagnostics["objective"]["finite"] is True
    variables = {item["name"]: item for item in diagnostics["variables"]}
    assert variables["device_on"]["raw_value"] == [0.25, 1.00002]
    assert variables["device_on"]["attributes"]["boolean"] is True
    assert variables["device_on"]["shape"] == [2]
    assert variables["power_kw"]["attributes"]["nonneg"] is True
    assert variables["power_kw"]["raw_value"] == [2.0, -0.25]
    assert diagnostics["max_integrality_residual"] == 0.25
    assert diagnostics["worst_integrality_variable_id"] == binary.id
    assert diagnostics["worst_integrality_location"] == [0]
    assert diagnostics["max_constraint_violation"] == 1.5
    first = diagnostics["constraints"][0]
    assert first["index"] == 0
    assert first["type"] == "Inequality"
    assert first["shape"] == [2]
    assert first["raw_value"] == [1.5, 0.0]
    assert first["raw_violation"] == [1.5, 0.0]
    assert first["raw_violation_max"] == 1.5
    assert first["violation_max"] == 1.5
    assert first["worst_location"] == [0]
    assert diagnostics["constraints"][1]["violation_max"] == 0.75
    assert diagnostics["native_solver"]["available"] is False
    for variable, value, attributes in zip(problem.variables(), before_values, before_attributes):
        np.testing.assert_array_equal(variable.value, value)
        assert variable.attributes == attributes
    assert problem.status is None
    assert problem.constraints == constraints
    json.dumps(diagnostics, allow_nan=False)


def test_missing_primals_and_constraint_errors_are_data():
    value = cp.Variable(name="unset")
    problem = cp.Problem(cp.Minimize(value), [value >= 0])

    diagnostics = collect_primal_diagnostics(problem)

    assert diagnostics["objective"]["missing"] is True
    assert diagnostics["objective"]["finite"] is False
    assert diagnostics["variables"][0]["missing"] is True
    constraint = diagnostics["constraints"][0]
    assert constraint["missing"] is True
    assert "ValueError" in constraint["errors"]["violation"]
    assert diagnostics["max_integrality_residual"] is None
    assert diagnostics["max_constraint_violation"] is None
    assert diagnostics["native_solver"]["variables"] == []
    assert value.value is None
    json.dumps(diagnostics, allow_nan=False)


def test_nonfinite_primals_remain_explicit_and_json_safe():
    value = cp.Variable(3, name="nonfinite")
    problem = cp.Problem(cp.Minimize(cp.sum(value)), [value <= 0])
    value.save_value(np.array([np.nan, np.inf, -np.inf]))

    diagnostics = collect_primal_diagnostics(problem)

    variable = diagnostics["variables"][0]
    assert variable["raw_value"] == ["NaN", "+Infinity", "-Infinity"]
    assert variable["finite"] is False
    assert variable["nonfinite_count"] == 3
    assert variable["nonfinite_locations"] == [[0], [1], [2]]
    assert diagnostics["objective"]["finite"] is False
    assert diagnostics["constraints"][0]["finite"] is False
    assert diagnostics["constraints"][0]["nonfinite_count"] == 2
    assert diagnostics["constraints"][0]["raw_violation_max"] == "NaN"
    assert diagnostics["constraints"][0]["raw_violation_worst_location"] == [0]
    np.testing.assert_array_equal(value.value, [np.nan, np.inf, -np.inf])
    json.dumps(diagnostics, allow_nan=False)


def test_indexed_integrality_ignores_continuous_coordinates():
    value = cp.Variable((2, 2), integer=[(0, 1)], name="partly_integer")
    problem = cp.Problem(cp.Minimize(cp.sum(value)), [value >= -1])
    value.save_value(np.array([[0.49, 1.01], [0.45, 0.4]]))

    diagnostics = collect_primal_diagnostics(problem)

    integrality = diagnostics["variables"][0]["integrality"]
    assert integrality["element_count"] == 1
    assert np.isclose(integrality["max_residual"], 0.01)
    assert integrality["worst_location"] == [0, 1]


class FakeNativeModel:
    Status = 9
    ObjVal = 1.75
    ObjBound = 1.5
    MIPGap = 1 / 7
    Runtime = 300.0
    MaxVio = 1.5
    BoundVio = 0.25
    ConstrVio = 1.5
    IntVio = 0.25

    def __init__(self, solution_count):
        self.SolCount = solution_count
        self.get_vars_calls = 0
        self.raw_variables = [SimpleNamespace(
            VarName="canonical_aux_17", X=-0.25, Obj=2.0,
            LB=0.0, UB=float("inf"), VType="C",
        )]

    def getVars(self):
        self.get_vars_calls += 1
        assert self.SolCount > 0
        return self.raw_variables


def native_problem(model):
    return SimpleNamespace(
        status="user_limit", objective=SimpleNamespace(value=1.75),
        variables=lambda: [], constraints=[],
        solver_stats=SimpleNamespace(extra_stats=model),
    )


def test_native_quality_and_raw_auxiliaries_are_read_without_mutation():
    model = FakeNativeModel(1)
    before = vars(model.raw_variables[0]).copy()

    diagnostics = collect_primal_diagnostics(native_problem(model))

    native = diagnostics["native_solver"]
    assert native["available"] is True
    assert native["has_incumbent"] is True
    assert native["attributes"]["MaxVio"] == 1.5
    assert native["attributes"]["BoundVio"] == 0.25
    assert native["attributes"]["ConstrVio"] == 1.5
    assert native["attributes"]["IntVio"] == 0.25
    assert "AttributeError" in native["attribute_errors"]["DualVio"]
    assert native["variables"] == [{
        "index": 0, "errors": {}, "VarName": "canonical_aux_17", "X": -0.25,
        "Obj": 2.0, "LB": 0.0, "UB": "+Infinity", "VType": "C",
    }]
    assert model.get_vars_calls == 1
    assert vars(model.raw_variables[0]) == before
    json.dumps(diagnostics, allow_nan=False)


def test_native_without_incumbent_never_reads_primal_variables():
    model = FakeNativeModel(0)

    diagnostics = collect_primal_diagnostics(native_problem(model))

    assert diagnostics["native_solver"]["has_incumbent"] is False
    assert diagnostics["native_solver"]["variables"] == []
    assert model.get_vars_calls == 0


def test_malformed_wrapper_failure_is_captured_without_affecting_dispatch():
    problem = SimpleNamespace(
        status="solver_error", objective=SimpleNamespace(value=None),
        variables=lambda: None, constraints=[], solver_stats=None,
    )

    diagnostics = collect_primal_diagnostics(problem)

    assert diagnostics["collection_failed"] is True
    assert "TypeError" in diagnostics["errors"][0]
    assert problem.status == "solver_error"
