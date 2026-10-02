"""Equal-probability scenario MPC, reusing the single-path formulations.

Only the first heat input is shared. Later inputs are scenario-specific recourse
decisions, and the controller applies only that shared first input before solving
again with new measurements.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

import cvxpy as cp
import numpy as np

if TYPE_CHECKING:
    from .water_heater import WaterHeaterPredictiveController


def build_scenario_problem(
    controller: WaterHeaterPredictiveController,
    optimization_kind: str,
    initial_state: np.ndarray,
    prices: np.ndarray,
    disturbances: np.ndarray,
    model: Mapping[str, Any] | None = None,
    *,
    num_scenarios: int | None = None,
    legionella_indicator: np.ndarray | None = None,
    **options: Any,
) -> tuple[cp.Problem, cp.Expression, cp.Expression | None]:
    """Return a problem and heat array shaped ``(scenario, horizon)``.

    Conditioning weights belong to forecast construction. Electricity,
    legionella, and persistence costs are equally averaged across paths.
    Comfort uses the maximum per-path total cost over the forecast horizon.
    """
    disturbances = np.asarray(disturbances, dtype=float)
    if (
        disturbances.ndim != 3
        or min(disturbances.shape) == 0
        or not np.isfinite(disturbances).all()
    ):
        raise ValueError(
            "Scenario disturbances must be finite with shape "
            "(n_scenarios, n_disturbances, horizon)"
        )
    count, _, steps = disturbances.shape
    if num_scenarios is not None and num_scenarios != count:
        raise ValueError("num_scenarios does not match scenario disturbances")
    if legionella_indicator is None:
        indicators = np.ones((count, steps))
    else:
        indicators = np.asarray(legionella_indicator, dtype=float)
        if indicators.shape == (steps,):
            indicators = np.broadcast_to(indicators, (count, steps))
        if indicators.shape != (count, steps):
            raise ValueError("Scenario indicators must have shape (n_scenarios, horizon)")
    if optimization_kind not in {"convex", "milp"}:
        raise ValueError("Unknown scenario optimization kind")

    constraints = []
    heat_paths = []
    temperature_paths = []
    components: dict[str, list[cp.Expression]] = {}
    for scenario in range(count):
        builder = (
            controller.optimization_problem_convex
            if optimization_kind == "convex"
            else controller.optimization_problem_milp
        )
        result = builder(
            initial_state, prices, disturbances[scenario], model,
            legionella_indicator=indicators[scenario], **options,
        )
        problem, heat = result[:2]
        constraints.extend(problem.constraints)
        if heat_paths:
            constraints.append(heat[0] == heat_paths[0][0])
        heat_paths.append(heat)
        if optimization_kind == "milp":
            temperature_paths.append(result[2])
        for name, expression in controller._objective_component_expressions.items():
            components.setdefault(name, []).append(expression)

    aggregated_components = {
        name: (
            cp.max(cp.hstack(expressions))
            if name == "comfort_objective_cost"
            else sum(expressions) / count
        )
        for name, expressions in components.items()
    }
    controller._objective_component_expressions = aggregated_components
    problem = cp.Problem(cp.Minimize(sum(aggregated_components.values())), constraints)
    return (
        problem,
        cp.vstack(heat_paths),
        cp.vstack(temperature_paths) if temperature_paths else None,
    )
