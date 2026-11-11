"""Grid 2 demand: maximum of complete, internally consistent scenario costs."""

import cvxpy as cp
import numpy as np

from .rates import GRID2_ANYTIME_DEMAND_USD_PER_KW_DAY, GRID2_ON_PEAK_DEMAND_USD_PER_KW_DAY

DEMAND_SCENARIO_AGGREGATION = "worst_complete_scenario_v1"


def scenario_demand_objective(block_power, day_masks, on_peak, observed_on,
                              observed_any, scenario_count, scale):
    """Keep each scenario intact across both components and all horizon days.

    Max expressions evaluate to the physical peaks, including in nonbinding
    scenarios; auxiliary epigraphs alone could have slack in those scenarios.
    CVXPY converts these convex piecewise-linear expressions to an LP/MILP.
    """
    peaks = {"on_peak": [], "anytime": []}
    for day, mask in enumerate(day_masks):
        for category, eligible, observed in (
            ("on_peak", mask & on_peak, observed_on[day]),
            ("anytime", mask, observed_any[day]),
        ):
            floor = np.full(scenario_count, max(0.0, observed))
            peak = (cp.maximum(floor, cp.max(block_power[:, eligible], axis=1))
                    if eligible.any() else cp.Constant(floor))
            peaks[category].append(peak)
    on = cp.vstack(peaks["on_peak"]).T if day_masks else None
    anytime = cp.vstack(peaks["anytime"]).T if day_masks else None
    on_cost = (scale * GRID2_ON_PEAK_DEMAND_USD_PER_KW_DAY * cp.sum(on, axis=1)
               if on is not None else cp.Constant(np.zeros(scenario_count)))
    any_cost = (scale * GRID2_ANYTIME_DEMAND_USD_PER_KW_DAY * cp.sum(anytime, axis=1)
                if anytime is not None else cp.Constant(np.zeros(scenario_count)))
    total = on_cost + any_cost
    return {
        "daily_demand_aggregation": DEMAND_SCENARIO_AGGREGATION,
        "daily_on_peak_demand_scenarios_kw": on,
        "daily_anytime_demand_scenarios_kw": anytime,
        # Retained summary fields are diagnostic envelopes, never added as costs.
        "daily_peak_summary_aggregation": "componentwise_max_for_diagnostics_only",
        "daily_on_peak_demand_kw": cp.max(on, axis=0) if on is not None else None,
        "daily_anytime_demand_kw": cp.max(anytime, axis=0) if anytime is not None else None,
        "daily_demand_on_peak_cost_scenarios": on_cost,
        "daily_demand_anytime_cost_scenarios": any_cost,
        "daily_demand_cost_scenarios": total,
        "daily_demand_objective_terms": {"worst_scenario": cp.max(total)},
    }


def demand_cost_metadata(plan, *, solved=True):
    """Report components from the same worst-cost scenario (first in a tie)."""
    keys = ("daily_on_peak_demand_scenarios_kw", "daily_anytime_demand_scenarios_kw",
            "daily_demand_on_peak_cost_scenarios", "daily_demand_anytime_cost_scenarios",
            "daily_demand_cost_scenarios")
    result = {key: (None if not solved or plan[key] is None
                    else np.asarray(plan[key].value).tolist()) for key in keys}
    worst = int(np.argmax(result["daily_demand_cost_scenarios"])) if solved else None
    result.update(
        daily_demand_aggregation=DEMAND_SCENARIO_AGGREGATION,
        daily_peak_summary_aggregation="componentwise_max_for_diagnostics_only",
        daily_demand_worst_scenario_index=worst,
        daily_demand_objective_terms={
            category: (None if not solved else result[f"daily_demand_{category}_cost_scenarios"][worst])
            for category in ("on_peak", "anytime")
        },
    )
    return result
