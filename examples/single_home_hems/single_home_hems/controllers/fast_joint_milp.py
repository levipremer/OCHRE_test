"""Exact sparse compiler for the joint controller's linear, piecewise-linear MILP.

The device builders remain the source of truth. This adapter evaluates their
CVXPY expression graph once per issue with memoized sparse affine operations,
uses standard abs/max epigraphs, and sends vector constraints to Gurobi. A native
model is reused only when its matrix sparsity and variable domains match. All
numeric coefficients, bounds, RHS values and objective entries are then updated;
there is no stale forecast or plant state and no solution warm start.

Unsupported atoms or nonlinear algebra raise UnsupportedFastMILP. The caller
may use the unchanged CVXPY solver for those problems. No term is approximated.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time
import weakref
from typing import Any

import cvxpy as cp
import cvxpy.settings as settings
from cvxpy.problems.problem import SolverStats
from cvxpy.reductions.solution import Solution
from cvxpy.reductions.solvers.qp_solvers.gurobi_qpif import GUROBI
import gurobipy as gp
import numpy as np
from scipy import sparse


class UnsupportedFastMILP(ValueError):
    """The graph contains algebra outside this explicitly supported adapter."""


def _array(value):
    if sparse.issparse(value):
        value = value.toarray()
    return np.asarray(value, dtype=float)


def _flat(value):
    return _array(value).reshape(-1, order="F")


def _broadcast_indices(shape, result_shape):
    return np.broadcast_to(np.arange(int(np.prod(shape or (1,)))).reshape(shape, order="F"),
                           result_shape).reshape(-1, order="F")


def _sum_map(shape, axis, keepdims):
    size = int(np.prod(shape or (1,)))
    if axis is None:
        return sparse.csr_matrix(np.ones((1, size)))
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    axes = tuple(a % len(shape) for a in axes)
    output_shape = tuple(1 if d in axes else n for d, n in enumerate(shape)) if keepdims else tuple(
        n for d, n in enumerate(shape) if d not in axes)
    coords = np.unravel_index(np.arange(size), shape, order="F")
    output_coords = tuple(np.zeros(size, dtype=int) if d in axes else coords[d]
                          for d in range(len(shape))) if keepdims else tuple(
        coords[d] for d in range(len(shape)) if d not in axes)
    rows = (np.ravel_multi_index(output_coords, output_shape, order="F")
            if output_shape else np.zeros(size, dtype=int))
    return sparse.csr_matrix((np.ones(size), (rows, np.arange(size))),
                             shape=(int(np.prod(output_shape or (1,))), size))


@dataclass
class Affine:
    matrix: sparse.csr_matrix
    offset: np.ndarray
    shape: tuple

    def rows(self, indices, shape):
        return Affine(self.matrix[indices], self.offset[indices], shape)

    def transform(self, matrix, shape):
        return Affine((matrix @ self.matrix).tocsr(), np.asarray(matrix @ self.offset).reshape(-1), shape)


@dataclass
class SparseMILP:
    matrix: sparse.csr_matrix
    rhs: np.ndarray
    sense: np.ndarray
    objective: np.ndarray
    objective_offset: float
    lower: np.ndarray
    upper: np.ndarray
    types: np.ndarray
    variables: list
    offsets: dict
    original_variable_count: int

    def topology_key(self):
        a = self.matrix
        digest = hashlib.sha256()
        for value in [np.asarray(a.shape, dtype=np.int64), a.indptr, a.indices,
                      self.types.astype("S1"), self.sense.astype("S1")]:
            digest.update(value.tobytes())
        return digest.hexdigest()

    def primal_check(self, z, tolerance=1e-3, integrality_tolerance=1e-4):
        activity = np.asarray(self.matrix @ z).reshape(-1)
        equal = self.sense == "="
        eq = float(np.max(np.abs(activity[equal]-self.rhs[equal]), initial=0))
        ine = float(np.max(activity[~equal]-self.rhs[~equal], initial=0))
        bounds = float(max(np.max(self.lower-z, initial=0), np.max(z-self.upper, initial=0)))
        integer = self.types != "C"
        integrality = float(np.max(np.abs(z[integer]-np.rint(z[integer])), initial=0))
        return dict(passed=max(eq, ine, bounds) <= tolerance and integrality <= integrality_tolerance,
                    equality_residual=eq, inequality_violation=max(0, ine),
                    bound_violation=max(0, bounds), integrality_violation=integrality)


class SparseGraphCompiler:
    """Memoized sparse evaluation, with convex epigraphs allocated up front."""
    PIECEWISE = {"abs", "maximum", "max", "norm1"}

    def __init__(self, problem):
        if not problem.is_dcp() or not isinstance(problem.objective, cp.Minimize):
            raise UnsupportedFastMILP("Only DCP minimization MILPs are supported")
        self.problem = problem
        self.variables = problem.variables()
        self.offsets, self.auxiliary = {}, {}
        self.n = 0
        for variable in self.variables:
            self.offsets[variable.id] = self.n
            self.n += variable.size
        self.original_n = self.n
        self.seen = set()
        self._allocate(problem.objective.expr)
        for constraint in problem.constraints:
            for arg in constraint.args:
                self._allocate(arg)
        self.memo = {}
        self.rows, self.rhs, self.senses = [], [], []

    def _allocate(self, expression):
        if id(expression) in self.seen:
            return
        self.seen.add(id(expression))
        if expression.is_constant():
            return
        for arg in expression.args:
            self._allocate(arg)
        kind = type(expression).__name__
        if kind in self.PIECEWISE:
            if kind in {"abs", "norm1"} and not expression.args[0].is_affine():
                raise UnsupportedFastMILP("abs and norm1 require affine arguments")
            count = expression.args[0].size if kind == "norm1" else expression.size
            self.auxiliary[id(expression)] = (self.n, count)
            self.n += count

    def _variable(self, start, size, shape):
        return Affine(sparse.csr_matrix((np.ones(size), (np.arange(size), start+np.arange(size))),
                                       shape=(size, self.n)), np.zeros(size), shape)

    def _constraint(self, expression, sense):
        self.rows.append(expression.matrix)
        self.rhs.append(-expression.offset)
        self.senses.extend([sense]*len(expression.offset))

    @staticmethod
    def _add(left, right, sign=1):
        return Affine((left.matrix + sign*right.matrix).tocsr(),
                      left.offset+sign*right.offset, left.shape)

    @staticmethod
    def _broadcast(value, shape):
        return value.rows(_broadcast_indices(value.shape, shape), shape)

    def expression(self, expression):
        key = id(expression)
        if key in self.memo:
            return self.memo[key]
        kind, shape = type(expression).__name__, expression.shape
        if expression.is_constant():
            value = _flat(expression.value)
            result = Affine(sparse.csr_matrix((expression.size, self.n)), value, shape)
        elif kind == "Variable":
            result = self._variable(self.offsets[expression.id], expression.size, shape)
        elif kind == "AddExpression":
            args = [self._broadcast(self.expression(a), shape) for a in expression.args]
            result = Affine(sum((a.matrix for a in args), sparse.csr_matrix((expression.size, self.n))),
                            sum((a.offset for a in args), np.zeros(expression.size)), shape)
        elif kind == "NegExpression":
            a = self.expression(expression.args[0])
            result = Affine(-a.matrix, -a.offset, shape)
        elif kind in {"multiply", "DivExpression"}:
            if kind == "DivExpression":
                if not expression.args[1].is_constant():
                    raise UnsupportedFastMILP("Division by a nonconstant expression")
                a = self._broadcast(self.expression(expression.args[0]), shape)
                weight = 1 / np.broadcast_to(_array(expression.args[1].value), shape).reshape(-1, order="F")
            else:
                left, right = expression.args
                if not left.is_constant():
                    left, right = right, left
                if not left.is_constant():
                    raise UnsupportedFastMILP("Product of two nonconstant expressions")
                a = self._broadcast(self.expression(right), shape)
                weight = np.broadcast_to(_array(left.value), shape).reshape(-1, order="F")
            result = Affine((sparse.diags(weight) @ a.matrix).tocsr(), weight*a.offset, shape)
        elif kind == "MulExpression":
            left, right = expression.args
            if left.is_constant():
                c = _array(left.value)
                a = self.expression(right)
                if c.ndim == 0:
                    transform = sparse.eye(a.matrix.shape[0], format="csr") * float(c)
                else:
                    c = c.reshape((1, -1)) if c.ndim == 1 else c
                    columns = 1 if len(right.shape) == 1 else right.shape[1]
                    transform = sparse.kron(sparse.eye(columns), sparse.csr_matrix(c), format="csr")
            elif right.is_constant():
                c = _array(right.value)
                a = self.expression(left)
                if c.ndim == 0:
                    transform = sparse.eye(a.matrix.shape[0], format="csr") * float(c)
                else:
                    c = c.reshape((-1, 1)) if c.ndim == 1 else c
                    rows = 1 if len(left.shape) == 1 else left.shape[0]
                    transform = sparse.kron(sparse.csr_matrix(c.T), sparse.eye(rows), format="csr")
            else:
                raise UnsupportedFastMILP("Matrix product of two nonconstant expressions")
            result = a.transform(transform, shape)
        elif kind in {"index", "special_index"}:
            a = self.expression(expression.args[0])
            indices = np.arange(expression.args[0].size).reshape(a.shape, order="F")[expression.key]
            result = a.rows(np.asarray(indices).reshape(-1, order="F"), shape)
        elif kind in {"reshape", "vec"}:
            a = self.expression(expression.args[0])
            indices = np.arange(expression.args[0].size).reshape(a.shape, order="F").reshape(
                shape, order=getattr(expression, "order", "F")).reshape(-1, order="F")
            result = a.rows(indices, shape)
        elif kind in {"Promote", "broadcast_to"}:
            result = self._broadcast(self.expression(expression.args[0]), shape)
        elif kind in {"Hstack", "Vstack"}:
            args = [self.expression(a) for a in expression.args]
            values, offset = [], 0
            for a in args:
                values.append(np.arange(offset, offset+len(a.offset)).reshape(a.shape, order="F"))
                offset += len(a.offset)
            combine = np.hstack if kind == "Hstack" else np.vstack
            indices = combine(values).reshape(-1, order="F")
            result = Affine(sparse.vstack([a.matrix for a in args], format="csr")[indices],
                            np.concatenate([a.offset for a in args])[indices], shape)
        elif kind == "Sum":
            a = self.expression(expression.args[0])
            result = a.transform(_sum_map(a.shape, expression.axis, expression.keepdims), shape)
        elif kind == "transpose":
            a = self.expression(expression.args[0])
            indices = np.arange(expression.args[0].size).reshape(a.shape, order="F").transpose(
                expression.axes).reshape(-1, order="F")
            result = a.rows(indices, shape)
        elif kind in self.PIECEWISE:
            start, count = self.auxiliary[key]
            auxiliary = self._variable(start, count, expression.args[0].shape if kind == "norm1" else shape)
            if kind in {"abs", "norm1"}:
                arg = self.expression(expression.args[0])
                self._constraint(self._add(arg, auxiliary, -1), "<")
                self._constraint(self._add(Affine(-arg.matrix, -arg.offset, arg.shape), auxiliary, -1), "<")
                result = (auxiliary.transform(_sum_map(arg.shape, expression.axis, expression.keepdims), shape)
                          if kind == "norm1" else auxiliary)
            elif kind == "maximum":
                for arg in expression.args:
                    self._constraint(self._add(self._broadcast(self.expression(arg), shape), auxiliary, -1), "<")
                result = auxiliary
            else:
                arg = self.expression(expression.args[0])
                # Broadcast the reduced max back to each entry of the argument.
                axes = expression.axis
                axes = tuple(range(len(arg.shape))) if axes is None else ((axes,) if isinstance(axes, int) else tuple(axes))
                axes = tuple(a % len(arg.shape) for a in axes)
                reduced = auxiliary
                expanded_shape = tuple(1 if d in axes else n for d, n in enumerate(arg.shape))
                reduced = Affine(reduced.matrix, reduced.offset, expanded_shape)
                self._constraint(self._add(arg, self._broadcast(reduced, arg.shape), -1), "<")
                result = auxiliary
        else:
            raise UnsupportedFastMILP(f"Unsupported expression atom {type(expression).__module__}.{kind}")
        if not np.isfinite(result.offset).all() or not np.isfinite(result.matrix.data).all():
            raise UnsupportedFastMILP("Nonfinite expression coefficient")
        self.memo[key] = result
        return result

    def compile(self):
        objective = self.expression(self.problem.objective.expr)
        original_rows = []
        for constraint in self.problem.constraints:
            kind = type(constraint).__name__
            if kind not in {"Equality", "Inequality"}:
                raise UnsupportedFastMILP(f"Unsupported constraint {kind}")
            left, right = constraint.args
            shape = np.broadcast_shapes(left.shape, right.shape)
            value = self._add(self._broadcast(self.expression(left), shape),
                              self._broadcast(self.expression(right), shape), -1)
            if kind == "Equality" and value.matrix[:, self.original_n:].nnz:
                raise UnsupportedFastMILP("A nonlinear epigraph appears in an equality")
            # A convex epigraph can only enter an original inequality with a
            # positive coefficient. Refuse algebra that needs a hypograph.
            if kind == "Inequality" and np.min(value.matrix[:, self.original_n:].data, initial=0) < -1e-14:
                raise UnsupportedFastMILP("A nonlinear epigraph requires a negative coefficient")
            original_rows.append((value, "=" if kind == "Equality" else "<"))
        if np.min(objective.matrix[:, self.original_n:].data, initial=0) < -1e-14:
            raise UnsupportedFastMILP("A nonlinear epigraph has a negative objective coefficient")
        for value, sense in original_rows:
            self._constraint(value, sense)
        matrix = sparse.vstack(self.rows, format="csr")
        matrix.sum_duplicates(); matrix.eliminate_zeros(); matrix.sort_indices()
        lower, upper = np.full(self.n, -np.inf), np.full(self.n, np.inf)
        types = np.full(self.n, "C", dtype="U1")
        for variable in self.variables:
            start = self.offsets[variable.id]
            section = slice(start, start+variable.size)
            attrs = variable.attributes
            unsupported = [name for name in ["complex", "imag", "symmetric", "diag", "PSD", "NSD", "hermitian", "sparsity"] if attrs.get(name)]
            if unsupported:
                raise UnsupportedFastMILP(f"Unsupported variable attributes {unsupported}")
            if attrs.get("nonneg") or attrs.get("pos"):
                lower[section] = 0
            if attrs.get("nonpos") or attrs.get("neg"):
                upper[section] = 0
            if attrs.get("bounds") is not None:
                lower[section], upper[section] = (_flat(a) for a in attrs["bounds"])
            for attr, domain in [("boolean", "B"), ("integer", "I")]:
                flag = attrs.get(attr)
                if flag is True:
                    indices = start+np.arange(variable.size)
                elif flag:
                    indices = start+np.ravel_multi_index(flag, variable.shape, order="F")
                else:
                    continue
                types[indices] = domain
                if domain == "B":
                    lower[indices], upper[indices] = 0, 1
        return SparseMILP(matrix, np.concatenate(self.rhs), np.asarray(self.senses),
                          objective.matrix.toarray().reshape(-1), float(objective.offset[0]),
                          lower, upper, types, self.variables, self.offsets, self.original_n)


class ReusableJointMILPSolver:
    """One native model, refreshed completely whenever a compatible issue arrives."""
    def __init__(self):
        self.model = self.variables = self.constraints = self.key = self.previous_matrix = None
        self.last_metrics = {}
        self.last_optimized_problem_ref = None

    def close(self):
        if self.model is not None:
            self.model.dispose()
        self.model = self.variables = self.constraints = self.key = self.previous_matrix = None
        self.last_optimized_problem_ref = None

    def solve(self, problem, options):
        # A failed compile must never make the retained previous model look
        # like a native candidate for the current issue.
        self.last_optimized_problem_ref = None
        self.last_metrics = {}
        started = time.perf_counter()
        data = SparseGraphCompiler(problem).compile()
        compiled = time.perf_counter()
        key = data.topology_key()
        reused = self.model is not None and key == self.key
        if not reused:
            self.close()
            self.model = gp.Model()
            self.model.Params.OutputFlag = 0
            self.variables = self.model.addMVar(len(data.objective), lb=data.lower, ub=data.upper, vtype=data.types)
            self.constraints = self.model.addMConstr(data.matrix, self.variables, data.sense, data.rhs)
            self.model.update()
            self.key = key
            self.previous_matrix = data.matrix.copy()
        else:
            self.model.reset()  # No incumbent, basis or solution warm start.
            self.variables.lb = data.lower
            self.variables.ub = data.upper
            self.constraints.RHS = data.rhs
            changed = np.flatnonzero(data.matrix.data != self.previous_matrix.data)
            if len(changed):
                rows = np.searchsorted(data.matrix.indptr[1:], changed, side="right")
                variables, constraints = self.variables.tolist(), self.constraints.tolist()
                for position, row in zip(changed, rows):
                    self.model.chgCoeff(constraints[row], variables[data.matrix.indices[position]],
                                        float(data.matrix.data[position]))
            self.previous_matrix = data.matrix.copy()
        # Keep the constant out of the native objective, matching CVXPY's
        # native relative-gap denominator and existing solver diagnostics.
        self.variables.Obj = data.objective
        self.model.ObjCon = 0.0
        self.model.ModelSense = gp.GRB.MINIMIZE
        self.model.resetParams()
        self.model.Params.OutputFlag = 0
        for name, value in options.items():
            self.model.setParam(name, value)
        built = time.perf_counter()
        self.model.optimize()
        native_done = time.perf_counter()
        self.last_optimized_problem_ref = weakref.ref(problem)
        self.last_metrics = dict(backend="sparse_native_v1", matrix_compile_seconds=compiled-started,
                                 native_model_update_seconds=built-compiled, native_model_reused=reused,
                                 solve_call_seconds=native_done-started,
                                 variable_count=len(data.objective), constraint_count=len(data.rhs),
                                 original_variable_count=data.original_variable_count,
                                 topology_key=key, primal_check=None,
                                 objective_constant_excluded_from_native=data.objective_offset,
                                 solution_warm_start=False)
        status = GUROBI.STATUS_MAP.get(self.model.Status, cp.SOLVER_ERROR)
        if status == cp.USER_LIMIT and not self.model.SolCount:
            status = cp.INFEASIBLE_INACCURATE
        if status == cp.SOLVER_ERROR:
            raise cp.error.SolverError(f"Native MILP returned solver status {self.model.Status}")
        attrs = {settings.SOLVE_TIME: self.model.Runtime, settings.NUM_ITERS: self.model.IterCount,
                 settings.EXTRA_STATS: self.model}
        primal, check = {}, None
        if self.model.SolCount:
            z = np.asarray(self.variables.X)
            check = data.primal_check(z)
            self.last_metrics["primal_check"] = check
            if not check["passed"]:
                raise cp.error.SolverError(f"Fast native MILP failed independent matrix feasibility: {check}")
            for variable in data.variables:
                start = data.offsets[variable.id]
                primal[variable.id] = z[start:start+variable.size].reshape(variable.shape, order="F")
            value = float(self.model.ObjVal+data.objective_offset)
        else:
            value = np.inf if status in {cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE} else -np.inf
        solution = Solution(status, value, primal, {}, attrs)
        problem.unpack(solution)
        problem._solver_stats = SolverStats.from_dict(attrs, "GUROBI")
        problem._compilation_time = compiled-started
        return data
